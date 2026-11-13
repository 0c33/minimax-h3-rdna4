"""Converts a ComfyUI UI workflow (nodes + links) to the API prompt format, using /object_info for input order.

Handles combos, control_after_generate, nested dynamic combos (key "a.b") and autogrow inputs (linked "values.a").
"""
import copy
import json
import urllib.request

BASE = "http://127.0.0.1:8188"
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}


def object_info():
    return json.load(urllib.request.urlopen(f"{BASE}/object_info", timeout=60))


def _is_widget(spec):
    t, opts = spec[0], (spec[1] if len(spec) > 1 else {})
    return (isinstance(t, list) or t in WIDGET_TYPES) and not opts.get("forceInput")


def _consume(inputs, values, linked, out, prefix=""):
    """Walks inputs (dict of required/optional) in order, taking widget values from the iterator."""
    for section in ("required", "optional"):
        for name, spec in inputs.get(section, {}).items():
            key = prefix + name
            if spec[0] == "COMFY_AUTOGROW_V3":
                for lname, src in linked.items():
                    if lname.startswith(key + "."):
                        out[lname] = src
                continue
            if not _is_widget(spec):
                if key in linked:
                    out[key] = linked[key]
                continue
            try:
                v = next(values)
            except StopIteration:
                if key in linked:
                    out[key] = linked[key]
                continue  # optional widgets missing from an older save keep their defaults
            out[key] = linked.get(key, v)
            opts = spec[1] if len(spec) > 1 else {}
            if opts.get("control_after_generate"):
                next(values, None)
            if spec[0] == "COMFY_DYNAMICCOMBO_V3":
                option = next(o for o in opts["options"] if o["key"] == v)
                _consume(option["inputs"], values, linked, out, prefix=key + ".")


def flatten(wf):
    """Inlines subgraph instances (one level). Returns (workflow, {(node id, input name): value}) where the second item
    holds the instance's promoted widget values, to be applied after conversion."""
    defs = {s["id"]: s for s in wf.get("definitions", {}).get("subgraphs", [])}
    if not any(n["type"] in defs for n in wf["nodes"]):
        return wf, {}
    wf = copy.deepcopy(wf)
    links = {l[0]: list(l) for l in wf["links"]}
    next_id = max(links, default=0) + 1
    nodes, overrides = [], {}
    for inst in wf["nodes"]:
        sg = defs.get(inst["type"])
        if sg is None:
            nodes.append(inst)
            continue
        if inst.get("mode", 0) in (2, 4):
            continue
        base = inst["id"] * 100000
        names = [i["name"] for i in sg["inputs"] if i["type"] in WIDGET_TYPES | {"COMBO"}]
        values = dict(zip(names, inst.get("widgets_values") or []))
        inst_links = {i["name"]: i.get("link") for i in inst.get("inputs", [])}
        inner = {}
        for n in sg["nodes"]:
            n = copy.deepcopy(n)
            n["id"] = base + n["id"]
            for i in n.get("inputs", []):
                i["link"] = None
            inner[n["id"]] = n
        for l in sg["links"]:
            o, os_, t, ts, ty = l["origin_id"], l["origin_slot"], l["target_id"], l["target_slot"], l["type"]
            if o == -10:
                name = sg["inputs"][os_]["name"]
                tgt = inner[base + t]
                if inst_links.get(name) is not None:
                    src = links[inst_links[name]]
                    links[next_id] = [next_id, src[1], src[2], base + t, ts, ty]
                    tgt["inputs"][ts]["link"] = next_id
                    next_id += 1
                elif name in values and "widget" in tgt["inputs"][ts]:
                    overrides[(str(base + t), tgt["inputs"][ts]["widget"]["name"])] = values[name]
            elif t == -20:
                for tl in links.values():
                    if tl[1] == inst["id"] and tl[2] == ts:
                        tl[1], tl[2] = base + o, os_
            else:
                links[next_id] = [next_id, base + o, os_, base + t, ts, ty]
                inner[base + t]["inputs"][ts]["link"] = next_id
                next_id += 1
        nodes.extend(inner.values())
    wf["nodes"], wf["links"] = nodes, list(links.values())
    return wf, overrides


def convert(wf, info=None):
    info = info or object_info()
    wf, overrides = flatten(wf)
    links = {l[0]: l for l in wf["links"]}
    prompt = {}
    for n in wf["nodes"]:
        if n["type"] in ("MarkdownNote", "Note") or n.get("mode", 0) in (2, 4):
            continue
        linked = {i["name"]: [str(links[i["link"]][1]), links[i["link"]][2]] for i in n.get("inputs", []) if i.get("link") is not None}
        out = {}
        _consume(info[n["type"]]["input"], iter(n.get("widgets_values") or []), linked, out)
        prompt[str(n["id"])] = {"class_type": n["type"], "inputs": out, "_meta": {"title": n.get("title") or n["type"]}}
    for (nid, name), value in overrides.items():
        if nid in prompt:
            prompt[nid]["inputs"][name] = value
    return prompt


def subset(prompt, outputs):
    """Keeps only the nodes the given output node ids depend on."""
    keep, todo = set(), [str(o) for o in outputs]
    while todo:
        nid = todo.pop()
        if nid in keep:
            continue
        keep.add(nid)
        todo += [v[0] for v in prompt[nid]["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)]
    return {k: v for k, v in prompt.items() if k in keep}
