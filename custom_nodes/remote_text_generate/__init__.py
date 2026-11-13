"""Text generation in ComfyUI runs on the text generation server only, never locally.

- Generate Text (Remote LLM): a node that calls the server directly, no text encoder needed.
- The core Generate Text nodes (TextGenerate, TextGenerateV2, TextGenerateLTX2Prompt) are rewired to the same
  server at startup, so official templates keep working. Their clip input becomes lazy, so a prompt-enhancer
  text encoder that only feeds them is never loaded. If the rewiring cannot apply, they refuse to run.

Images, videos and audio reach the server only when its model accepts them; a text-only model gets the text alone.

Environment (API keys never go in widgets, the workflow is saved into every PNG):
  REMOTE_LLM_URL      server for text generation       default http://127.0.0.1:8000/v1
  REMOTE_LLM_MODEL    model id or alias                default: first model the server lists
  REMOTE_LLM_API_KEY  bearer token for REMOTE_LLM_URL  optional
"""
import base64
import io as pyio
import json
import logging
import os
import sys

import numpy as np
import requests
from PIL import Image
from typing_extensions import override

import comfy.model_management
import nodes
from comfy_api.latest import ComfyExtension, io

SERVER_URL = os.environ.get("REMOTE_LLM_URL", "http://127.0.0.1:8000/v1")
SERVER_MODEL = os.environ.get("REMOTE_LLM_MODEL", "")


def _headers():
    key = os.environ.get("REMOTE_LLM_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _pick_model(base, model):
    """Returns (model id, input modalities) of the requested model, or of the first one the server lists."""
    try:
        r = requests.get(f"{base}/models", headers=_headers(), timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"text generation server {base} is not reachable ({e}). Local text generation is disabled.") from e
    entries = r.json().get("data", [])
    if not entries:
        raise RuntimeError(f"{base} lists no models")
    entry = next((e for e in entries if e.get("id") == model or model in e.get("aliases", [])), None) if model else entries[0]
    if entry is None:
        raise RuntimeError(f"model '{model}' not found on {base}, it serves: {[e.get('id') for e in entries]}")
    return entry["id"], entry.get("architecture", {}).get("input_modalities", ["text"])


def _image_url(image, max_side=1536):
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = pyio.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _stream_chat(base, body):
    text, reasoning = [], []
    with requests.post(f"{base}/chat/completions", json=body, headers=_headers(), stream=True, timeout=(15, 900)) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{base} returned {r.status_code}: {r.text[:500]}")
        for raw in r.iter_lines():
            comfy.model_management.throw_exception_if_processing_interrupted()
            line = raw.decode("utf-8")  # event streams carry no charset, requests would guess latin-1
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            reasoning.append(delta.get("reasoning_content") or "")
            text.append(delta.get("content") or "")
    text, reasoning = "".join(text), "".join(reasoning)
    # servers that don't parse reasoning leave the think block in the content
    if "</think>" in text:
        think, _, text = text.partition("</think>")
        reasoning = reasoning or think.replace("<think>", "", 1)
    return text.strip(), reasoning.strip()


def remote_generate(prompt, system_prompt, sampling_mode, max_length, thinking, images=(), server_url=SERVER_URL, model=SERVER_MODEL):
    """Runs one chat completion on the server. images are [H,W,C] tensors."""
    base = server_url.rstrip("/")
    model_id, modalities = _pick_model(base, model)
    content = prompt
    if images and "image" in modalities:
        content = [{"type": "image_url", "image_url": {"url": _image_url(img)}} for img in images] + [{"type": "text", "text": prompt}]
    elif images:
        logging.info("Remote text generation: %s reads text only, the %d image(s) are not sent.", model_id, len(images))

    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": content}]
    body = {"model": model_id, "messages": messages, "max_tokens": max_length, "stream": True,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    if sampling_mode.get("sampling_mode") == "on":
        body.update(temperature=sampling_mode.get("temperature", 1.0), top_k=sampling_mode.get("top_k", 50), top_p=sampling_mode.get("top_p", 1.0),
                    min_p=sampling_mode.get("min_p", 0.0), repeat_penalty=sampling_mode.get("repetition_penalty", 1.0),
                    presence_penalty=sampling_mode.get("presence_penalty", 0.0), seed=(sampling_mode.get("seed") or 0) % 2**32)
    else:
        body.update(temperature=0.0, top_k=1)

    text, reasoning = _stream_chat(base, body)
    logging.info("Remote text generation: %s on %s wrote %d chars (%d reasoning).", model_id, base, len(text), len(reasoning))
    return text, reasoning


class RemoteTextGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("repetition_penalty", default=1.0, min=0.0, max=5.0, step=0.01),
                    io.Float.Input("presence_penalty", default=0.0, min=0.0, max=5.0, step=0.01),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                ]
            ),
            io.DynamicCombo.Option(key="off", inputs=[]),
        ]
        return io.Schema(
            node_id="RemoteTextGenerate",
            display_name="Generate Text (Remote LLM)",
            category="text",
            search_aliases=["LLM", "prompt enhance", "openai", "llama.cpp", "remote"],
            description="Generates text on the text generation server. Images are sent only when the server's model accepts them.",
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplateNames(io.Image.Input("image"), names=[f"image{i}" for i in range(1, 17)], min=0)),
                io.Int.Input("max_length", default=4096, min=1, max=131072),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
                io.Boolean.Input("thinking", optional=True, default=False, tooltip="Ask the model to reason before answering (chat template enable_thinking)."),
                io.String.Input("system_prompt", force_input=True, optional=True),
                io.String.Input("server_url", default=SERVER_URL, advanced=True, tooltip="OpenAI-compatible base URL. Key: REMOTE_LLM_API_KEY env var."),
                io.String.Input("model", default=SERVER_MODEL, advanced=True, tooltip="Model id or alias. Empty uses the first model the server lists."),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
                io.String.Output(display_name="thinking"),
            ],
        )

    @classmethod
    def execute(cls, prompt, max_length, sampling_mode, thinking=False, system_prompt="", server_url=SERVER_URL, model=SERVER_MODEL, images: io.Autogrow.Type = None) -> io.NodeOutput:
        images = [batch[i] for batch in (images or {}).values() if batch is not None for i in range(batch.shape[0])]
        text, reasoning = remote_generate(prompt, system_prompt, sampling_mode, max_length, thinking, images=images, server_url=server_url, model=model)
        return io.NodeOutput(text or prompt, reasoning)


# --- the core Generate Text nodes, rewired to the server ---

def _images(image=None, images=None):
    return [b[i] for b in [image] + list((images or {}).values()) if b is not None for i in range(b.shape[0])]


# clip is optional: unconnected, ComfyUI does not pass it at all
def _remote_textgen_execute(cls, prompt, max_length, sampling_mode, clip=None, image=None, thinking=False, use_default_template=True, video=None, audio=None, mtp="auto", system_prompt="", images=None, videos=None, audios=None):
    if video is not None or audio is not None or videos or audios:
        logging.info("Remote text generation: video and audio inputs are not sent to the text generation server.")
    text, reasoning = remote_generate(prompt, system_prompt if use_default_template else "", sampling_mode, max_length, thinking, images=_images(image, images))
    return io.NodeOutput(text, reasoning)


def _remote_ltx2_execute(cls, prompt, max_length, sampling_mode, clip=None, image=None, thinking=False, use_default_template=True, video=None, audio=None, mtp="auto", system_prompt=""):
    core = sys.modules[cls.__module__]
    if "gemma4" in getattr(getattr(clip, "tokenizer", None), "clip_name", ""):
        system = system_prompt.strip() or (core.LTX24_I2V_SYSTEM_PROMPT if image is not None else core.LTX24_T2V_SYSTEM_PROMPT).strip()
        user = f"User Raw Input Prompt: {prompt}." if image is not None else f"user prompt: {prompt}"
    else:
        system = system_prompt.strip() or (core.LTX2_I2V_SYSTEM_PROMPT if image is not None else core.LTX2_T2V_SYSTEM_PROMPT).strip()
        user = f"User Raw Input Prompt: {prompt}."
    text, reasoning = remote_generate(user, system, sampling_mode, max_length, thinking, images=_images(image))
    return io.NodeOutput(text or prompt, reasoning)


def _local_textgen_disabled(cls, **kwargs):
    raise RuntimeError(f"{cls.__name__}: local text generation is disabled and the remote rewiring did not apply to this ComfyUI version. "
                       "Use the Generate Text (Remote LLM) node or update custom_nodes/remote_text_generate.")


def _rewire(node_id, execute, clip_needed):
    cls = nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if cls is None:
        return
    try:
        define_schema = cls.__dict__["define_schema"].__func__

        def lazy_clip_schema(klass):
            schema = define_schema(klass)
            for inp in schema.inputs:
                if inp.id == "clip":
                    inp.lazy, inp.optional = True, True
            return schema

        cls.define_schema = classmethod(lazy_clip_schema)
        cls.check_lazy_status = classmethod(lambda klass, clip=None, **kwargs: ["clip"] if clip_needed and clip is None else [])
        cls.execute = classmethod(execute)
        cls.GET_SCHEMA()
    except Exception as e:
        cls.execute = classmethod(_local_textgen_disabled)
        logging.error("remote_text_generate: could not rewire %s (%s), it is disabled instead.", node_id, e)


class RemoteTextGenerateExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        _rewire("TextGenerate", _remote_textgen_execute, clip_needed=False)
        _rewire("TextGenerateV2", _remote_textgen_execute, clip_needed=False)
        # LTX2 picks its system prompt from the text encoder name, and the workflow loads that encoder anyway
        _rewire("TextGenerateLTX2Prompt", _remote_ltx2_execute, clip_needed=True)
        logging.info("remote_text_generate: all Generate Text nodes run on %s (model: %s).", SERVER_URL, SERVER_MODEL or "server default")

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [RemoteTextGenerate]


async def comfy_entrypoint() -> RemoteTextGenerateExtension:
    return RemoteTextGenerateExtension()
