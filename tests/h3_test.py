"""End-to-end test of the MiniMax-H3 workflows on this PC, through the running ComfyUI (http://127.0.0.1:8188).

    .venv/bin/python user/h3_tests/h3_test.py t2v                  # text to video, 5 s, 20 steps, remote prompt writer
    .venv/bin/python user/h3_tests/h3_test.py i2v --image foo.png  # image to video (file in ComfyUI/input)
    options: --duration S  --steps N  --turbo  --no-writer  --idea "text"  --seed N  --tag NAME

Runs the saved workflow exactly as the UI would (only the options above are changed), records GPU busy, VRAM, RAM,
swap, disk reads and drive temperature 4x per second, then checks the result:
  - the job finished, and the log has no error, traceback or out-of-memory
  - the video has the expected size, frame count and 24 fps, and no black, frozen or NaN frames
  - the soundtrack covers the clip and is not silent
  - with the writer on: the prompt H3 got is in H3's format (and starts with the reference line for image to video)
  - RAM never dropped below 1 GB available
Writes runs/<tag>.json (all numbers), runs/<tag>.log (ComfyUI log of the run) and runs/<tag>_frames.jpg (contact sheet).
Exit code 0 = all checks passed.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from comfy_api import run  # noqa: E402
from ui2api import convert  # noqa: E402

COMFY = HERE.parents[1]
WORKFLOWS = {"t2v": "MiniMax-H3 bf16 - Text to Video.json", "i2v": "MiniMax-H3 bf16 - Image to Video.json"}
LOG = COMFY / "user/run_comfy.log"
CARD = next(p.parent for p in Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"))
# temperature of the first NVMe drive (set H3_TEST_NVME_MODEL to pick another one by model name)
NVME = next((h for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")) if (h / "name").read_text().strip() == "nvme"
             and os.environ.get("H3_TEST_NVME_MODEL", "") in (h / "device/model").read_text()), None)
# first match of each name starts that phase; the writer finishing is where the text encoder starts
PHASES = [("prompt writer", r"got prompt"), ("text encoder", r"Remote text generation: .* wrote"),
          ("text encoder", r"Requested to load MiniMaxH3TEModel_"), ("sampling", r"Requested to load MiniMaxH3$"),
          ("audio decode", r"Requested to load MiniMaxH3AudioVAE"), ("video decode", r"Requested to load MiniMaxH3VideoVAE"),
          ("done", r"Prompt executed in")]
BAD_LOG = re.compile(r"Traceback|[Oo]ut of memory|OutOfMemory|HIP error|CUDA error|\bNaN\b|invalid value encountered|disabled after an error")


def frame_count(seconds):
    n = max(5, round(seconds * 24))
    return n + (5 - n % 17) % 17


def comfy_pid():
    for p in Path("/proc").iterdir():
        try:
            if p.name.isdigit() and b"main.py\0--listen" in (p / "cmdline").read_bytes():
                return int(p.name)
        except OSError:
            pass
    raise SystemExit("ComfyUI is not running")


def build_prompt(a):
    prompt = convert(json.loads((COMFY / "user/default/workflows" / WORKFLOWS[a.mode]).read_text()))
    for nid, v in prompt.items():
        ct, title = v["class_type"], v["_meta"]["title"]
        if ct == "PrimitiveFloat" and title == "Float (duration)":
            v["inputs"]["value"] = a.duration
        elif ct == "SaveVideo":
            v["inputs"]["filename_prefix"] = f"video/test_{a.tag}"
        elif ct == "PrimitiveBoolean" and "Lightning" in title:
            v["inputs"]["value"] = a.turbo
        elif ct == "PrimitiveBoolean" and title.startswith("Use remote prompt writer"):
            v["inputs"]["value"] = not a.no_writer
        elif ct == "PrimitiveBoolean" and title.startswith("Fast mode"):
            v["inputs"]["value"] = a.fast
        elif ct == "RandomNoise" and a.seed is not None:
            v["inputs"]["noise_seed"] = a.seed
        elif ct == "LoadImage" and a.image:
            v["inputs"]["image"] = a.image
        elif nid == "2001" and a.idea:
            v["inputs"]["value"] = a.idea
    if a.steps:
        steps = next(v for v in prompt.values() if v["class_type"] == "ComfySwitchNode" and v["_meta"]["title"] == "If/Else Switch (Steps)")
        prompt[steps["inputs"]["on_true" if a.turbo else "on_false"][0]]["inputs"]["value"] = a.steps
    # optional speed patches between the model and the sampler (both approximate: they change the output)
    model_src = next(k for k, v in prompt.items() if v["class_type"] == "ComfySwitchNode" and v["_meta"]["title"] == "If/Else Switch (Model)")
    src = [model_src, 0]
    for i, patch in enumerate(a.patch or []):
        name, _, args = patch.partition(":")
        opts = dict(kv.split("=") for kv in args.split(",") if kv)
        if name == "sparse":
            method = opts.pop("method", "sol-attn")
            node = {"class_type": "BlockSparseAttention", "inputs": {
                "model": src, "selection": method, "start_percent": float(opts.pop("start", 0.2)), "end_percent": float(opts.pop("end", 1.0)),
                "dense_blocks": opts.pop("dense", ""), "min_tokens": 12288, "extra_tokens": int(opts.pop("extra", 256)),
                "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
            node["inputs"]["selection.tau" if method == "sol-attn" else "selection.keep_percent"] = float(opts.pop("tau" if method == "sol-attn" else "keep", 1.3 if method == "sol-attn" else 10.0))
        elif name == "easycache":
            node = {"class_type": "EasyCache", "inputs": {"model": src, "reuse_threshold": float(opts.pop("threshold", 0.2)),
                                                          "start_percent": float(opts.pop("start", 0.15)), "end_percent": float(opts.pop("end", 0.95)), "verbose": False}}
        else:
            raise SystemExit(f"unknown patch {name}")
        if opts:
            raise SystemExit(f"unknown options for {name}: {opts}")
        nid = str(9000 + i)
        prompt[nid] = dict(node, _meta={"title": patch})
        src = [nid, 0]
    for v in prompt.values():
        if v["class_type"] in ("BasicGuider", "BasicScheduler") and v["inputs"].get("model") == [model_src, 0]:
            v["inputs"]["model"] = src
    return prompt


class Monitor(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.samples, self.events, self.steps, self.stop = pid, [], [], {}, threading.Event()
        self.pos = self.log_start = LOG.stat().st_size

    def run(self):
        buf = ""
        while not self.stop.is_set():
            t = time.time()
            m = {k: int(v.split()[0]) / 2**20 for k, v in (l.split(":", 1) for l in open("/proc/meminfo"))}
            rchar = next(int(l.split()[1]) for l in open(f"/proc/{self.pid}/io") if l.startswith("rchar"))
            self.samples.append(dict(t=t, busy=int((CARD / "gpu_busy_percent").read_text()), vram=int((CARD / "mem_info_vram_used").read_text()) / 2**30,
                                     avail=m["MemAvailable"], swap=m["SwapTotal"] - m["SwapFree"], rchar=rchar,
                                     nvme=int((NVME / "temp1_input").read_text()) / 1000 if NVME else 0))
            with open(LOG, "rb") as f:
                f.seek(self.pos)
                chunk = f.read()
            self.pos += len(chunk)
            buf += re.sub(r"\x1b\[[0-9;]*m", "", chunk.decode("utf-8", "replace"))
            *lines, buf = re.split(r"[\r\n]", buf)
            for line in lines:
                for name, pat in PHASES:
                    if re.search(pat, line):
                        self.events.append((t, name))
                mm = re.search(r"\|\s*(\d+)/(\d+) \[", line)
                if mm and int(mm.group(1)) not in self.steps:
                    self.steps[int(mm.group(1))] = t
                    # loaded models print no "Requested to load", so the progress bar marks sampling and decoding too
                    if mm.group(1) == "0":
                        self.events.append((t, "sampling"))
                    elif mm.group(1) == mm.group(2):
                        self.events.append((t, "decode"))
            self.stop.wait(0.25)

    def window(self, ta, tb):
        s = [x for x in self.samples if ta <= x["t"] <= tb]
        if len(s) < 2:
            return None
        return dict(seconds=round(tb - ta, 1), disk_gb=round((s[-1]["rchar"] - s[0]["rchar"]) / 1e9, 1),
                    gpu_busy=round(sum(x["busy"] for x in s) / len(s)), vram_max=round(max(x["vram"] for x in s), 2),
                    ram_avail_min=round(min(x["avail"] for x in s), 2), swap_max=round(max(x["swap"] for x in s), 2),
                    nvme_max_c=max(x["nvme"] for x in s))


def check_video(path, seconds):
    """Returns (facts, problems) for the saved mp4."""
    problems, facts = [], {}
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        w, h = vs.codec_context.width, vs.codec_context.height
        # quarter-size frames are enough to judge exposure and motion, and keep a 15 s clip under 100 MB of RAM
        frames = [np.asarray(f.to_image().resize((w // 4, h // 4), Image.BILINEAR)) for f in c.decode(vs)]
    facts.update(width=w, height=h, frames=len(frames), fps=float(vs.average_rate))
    if len(frames) != frame_count(seconds):
        problems.append(f"{len(frames)} frames, expected {frame_count(seconds)}")
    if round(facts["fps"]) != 24:
        problems.append(f"fps {facts['fps']}")
    arr = np.stack(frames[::max(1, len(frames) // 24)]).astype(np.float32)
    means, stds = arr.mean(axis=(1, 2, 3)), arr.std(axis=(1, 2, 3))
    motion = float(np.abs(np.diff(arr, axis=0)).mean()) if len(arr) > 1 else 0.0
    facts.update(brightness_min=round(float(means.min()), 1), contrast_min=round(float(stds.min()), 1), motion=round(motion, 2))
    if means.min() < 4 or stds.min() < 2:
        problems.append("black or flat frames")
    if motion < 0.3:
        problems.append("the video is frozen")
    with av.open(str(path)) as c:
        if not c.streams.audio:
            problems.append("no soundtrack")
        else:
            a = c.streams.audio[0]
            pcm = np.concatenate([f.to_ndarray().reshape(-1) for f in c.decode(a)]).astype(np.float32)
            dur = pcm.size / a.channels / a.rate
            rms = float(np.sqrt(np.mean(pcm ** 2)))
            facts.update(audio_seconds=round(dur, 2), audio_rate=a.rate, audio_channels=a.channels, audio_rms_db=round(20 * np.log10(max(rms, 1e-9)), 1))
            if dur < seconds * 0.95:
                problems.append(f"soundtrack is {dur:.2f} s")
            if rms < 1e-3:
                problems.append("soundtrack is silent")
    pick = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, 8)]
    thumbs = [Image.fromarray(f) for f in pick]
    sheet = Image.new("RGB", (thumbs[0].width * 4, thumbs[0].height * 2))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % 4) * t.width, (i // 4) * t.height))
    return facts, problems, sheet


def check_prompt(text, mode):
    problems = []
    for field in ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:"):
        if field not in text:
            problems.append(f"prompt lacks {field}")
    if mode == "i2v" and not text.startswith("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."):
        problems.append("image-to-video prompt does not start with the <Picture 1> reference line")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=WORKFLOWS)
    ap.add_argument("--duration", type=float, default=5)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--turbo", action="store_true")
    ap.add_argument("--no-writer", action="store_true")
    ap.add_argument("--fast", action="store_true", help="turn on the workflow's fast mode (EasyCache)")
    ap.add_argument("--idea")
    ap.add_argument("--image")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--patch", action="append", help="approximate speed-up between model and sampler: "
                    "sparse[:method=sol-attn|sla|vsa,tau=1.3,keep=10,start=0.2,end=1.0,dense=0-1,extra=256] or easycache[:threshold=0.2,start=0.15,end=0.95]")
    a = ap.parse_args()
    a.tag = a.tag or time.strftime(f"{a.mode}_{a.duration:g}s_%Y%m%d_%H%M%S")
    out = HERE / "runs"
    out.mkdir(exist_ok=True)

    prompt = build_prompt(a)
    mon = Monitor(comfy_pid())
    mon.start()
    hist, wall, status = run(prompt, timeout=6 * 3600)
    time.sleep(1)
    mon.stop.set()
    mon.join()
    log = re.sub(r"\x1b\[[0-9;]*m", "", LOG.read_bytes()[mon.log_start:].decode("utf-8", "replace"))
    (out / f"{a.tag}.log").write_text(log)

    report = dict(tag=a.tag, mode=a.mode, fast=a.fast, patches=a.patch or [], duration=a.duration, steps=a.steps or (8 if a.turbo else 20), turbo=a.turbo,
                  writer=not a.no_writer, status=status, wall_seconds=round(wall), phases={}, step_seconds=[], problems=[])
    if status != "success":
        report["problems"].append(f"job status {status}: {json.dumps(hist.get('status'))[-1500:]}")
    first = {}
    for t, name in mon.events:
        first.setdefault(name, t)
    marks = sorted((t, n) for n, t in first.items()) + [(mon.samples[-1]["t"], "end")]
    for (ta, na), (tb, _) in zip(marks, marks[1:]):
        if na != "done" and (w := mon.window(ta, tb)):
            report["phases"][na] = w
    ks = sorted(mon.steps)
    report["step_seconds"] = [round(mon.steps[b] - mon.steps[c], 1) for c, b in zip(ks, ks[1:]) if mon.steps[b] > mon.steps[c]]
    report["whole_run"] = mon.window(mon.samples[0]["t"], mon.samples[-1]["t"])
    if report["whole_run"]["ram_avail_min"] < 1.0:
        report["problems"].append(f"RAM available fell to {report['whole_run']['ram_avail_min']} GB")
    for line in log.splitlines():
        if BAD_LOG.search(line):
            report["problems"].append("log: " + line.strip()[:200])

    outputs = hist.get("outputs", {})
    final_prompt = next((o["text"][0] for nid, o in outputs.items() if nid == "2006" and o.get("text")), None)
    report["final_prompt"] = final_prompt
    if not a.no_writer:
        report["problems"] += check_prompt(final_prompt or "", a.mode)
    video = next((f for o in outputs.values() for v in o.values() if isinstance(v, list) for f in v
                  if isinstance(f, dict) and str(f.get("filename", "")).endswith(".mp4")), None)
    if video is None:
        report["problems"].append("no video was saved")
    else:
        path = COMFY / "output" / video["subfolder"] / video["filename"]
        report["video"] = str(path)
        facts, problems, sheet = check_video(path, a.duration)
        report["video_facts"] = facts
        report["problems"] += problems
        sheet.save(out / f"{a.tag}_frames.jpg", quality=88)

    (out / f"{a.tag}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "final_prompt"}, indent=1))
    print("PASS" if not report["problems"] else "FAIL")
    sys.exit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
