#!/usr/bin/env python3
"""Drive a chunked take through ComfyUI headlessly: validate, submit, poll, report.

WHY THIS EXISTS
  Every piece of the long-form path is tested in isolation and the whole path
  had only ever run with a browser attached. `H3ChunkClose` REWRITES THE GRAPH
  at execution time -- it clones the body once per chunk through ComfyUI's
  dynamic-expansion API -- and that had never been exercised by a bare POST to
  /prompt. This is the script that proves it does, and the one a front end's
  render call will be built on.

WHAT IT CHECKS BEFORE SPENDING GPU TIME
  An API-format graph is a dict of node ids, and the two ways it rots are
  silent: a class_type the server does not have, and a COMBO value that is not
  in the server's option list. The second is the one that bit us -- an empty
  string in a LoRA picker fails PROMPT VALIDATION with an error naming stale
  labels. Both are checked against the live /object_info first, so a bad graph
  costs a second rather than a queue slot.

USAGE
    python3 run_take.py take_API.json                     # validate only
    python3 run_take.py take_API.json --submit            # run the whole plan
    python3 run_take.py take_API.json --submit --chunks 1 # just the first chunk
    python3 run_take.py take_API.json --submit --watch    # ... and wait for it

  --chunks N renders only the first N chunks by setting H3ChunkOpen's
  first_chunk/last_chunk, which is how you prove the mechanism for the price of
  one chunk instead of a whole take.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "http://127.0.0.1:8188"


def api(host, path, payload=None, timeout=30):
    url = host.rstrip("/") + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def load_graph(path):
    g = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if "nodes" in g and "links" in g:
        sys.exit(f"{path} is a GUI workflow, not an API export. In ComfyUI: "
                 f"Workflow -> Export (API).")
    bad = [k for k, v in g.items() if not isinstance(v, dict)
           or "class_type" not in v]
    if bad:
        sys.exit(f"{path}: {len(bad)} entr(ies) are not API nodes: {bad[:4]}")
    return g


def validate(host, graph):
    """-> (problems, notes). Problems block; notes are for reading.

    THE SPLIT MATTERS. A COMBO value the server will not accept is a hard stop
    -- that is the failure that cost a render, an empty string in a LoRA picker
    rejected at prompt validation. An input the server does not DECLARE is not:
    the frontend writes extras that ComfyUI tolerates, including dotted
    sub-paths of a COMFY_DYNAMICCOMBO_V3 (`format.codec`,
    `format.codec.encoding`) and UI-only keys like `video-preview`. Blocking on
    those rejected a graph that renders perfectly well.
    """
    info = api(host, "/object_info")
    problems, notes = [], []
    for nid, node in sorted(graph.items(), key=lambda kv: int(kv[0])):
        ct = node["class_type"]
        spec = info.get(ct)
        if spec is None:
            problems.append(f"{ct}#{nid}: the server has no such node")
            continue
        declared = {}
        for grp in ("required", "optional"):
            declared.update((spec.get("input") or {}).get(grp) or {})
        for name, val in (node.get("inputs") or {}).items():
            if isinstance(val, list):
                continue                      # a link, checked by the server
            decl = declared.get(name)
            if decl is None:
                # a dotted sub-path belongs to its prefix; anything else is a
                # frontend extra the server ignores
                root = name.split(".", 1)[0]
                if root not in declared:
                    notes.append(f"{ct}#{nid}: `{name}` is not declared "
                                 f"(frontend extra, ignored by the server)")
                continue
            opts = decl[0]
            if isinstance(opts, list) and val not in opts:
                shown = ", ".join(map(str, opts[:4]))
                problems.append(
                    f"{ct}#{nid}: `{name}` = {val!r} is not an option "
                    f"({len(opts)} available: {shown}...)")
        for name, decl in declared.items():
            cfg = decl[1] if len(decl) > 1 and isinstance(decl[1], dict) else {}
            if name in (node.get("inputs") or {}):
                continue
            if "default" in cfg or isinstance(decl[0], list):
                continue                      # server fills it
    return problems, notes


def chunk_limit(graph, n):
    """Shorten the take to n chunks by lowering chunk_count. -> what changed.

    NOT via H3ChunkOpen's first_chunk/last_chunk gate. Those indices are
    0-BASED and `last_chunk = 0` means "to the end", so the gate cannot express
    "only the first chunk" at all -- (0, 1) is the first TWO. The gate is for
    resuming a take that was stopped; shortening one is the plan's job.

    Two chunks is the cheapest honest proof of the chunked path, because the
    expansion in H3ChunkClose only clones anything when there is more than one.
    """
    touched = []
    for nid, node in graph.items():
        if node["class_type"] != "H3ChunkPlan":
            continue
        ins = node.setdefault("inputs", {})
        if isinstance(ins.get("chunk_count"), list):
            touched.append(f"H3ChunkPlan#{nid}: chunk_count is WIRED, "
                           f"--chunks ignored")
            continue
        was = ins.get("chunk_count")
        ins["chunk_count"] = int(n)
        touched.append(f"H3ChunkPlan#{nid}: chunk_count {was} -> {n}")
        if int(n) < 2:
            touched.append("  NOTE one chunk does not exercise the graph "
                           "expansion — use 2 to prove the chunked path")
    return touched


def plan_of(host, graph):
    """Ask the planner what this graph will produce, before running it."""
    for node in graph.values():
        if node["class_type"] != "H3ChunkPlan":
            continue
        ins = node.get("inputs") or {}
        if any(isinstance(ins.get(k), list)
               for k in ("chunk_frames", "chunk_count", "context")):
            return None
        try:
            return api(host, "/h3_toolkit/chunk/plan", {
                "chunk_frames": ins.get("chunk_frames", 141),
                "chunk_count": ins.get("chunk_count", 4),
                "context": int(ins.get("context", 39)),
            })
        except (urllib.error.URLError, urllib.error.HTTPError):
            return None
    return None


def submit(host, graph):
    out = api(host, "/prompt", {"prompt": graph, "client_id": "run_take"})
    if "prompt_id" not in out:
        sys.exit("submit refused:\n" + json.dumps(out, indent=2)[:2000])
    return out["prompt_id"]


def watch(host, pid, poll=5.0):
    """Block until the prompt leaves the queue. -> (seconds, history entry)."""
    t0 = time.time()
    last = ""
    while True:
        hist = api(host, f"/history/{pid}")
        if pid in hist:
            return time.time() - t0, hist[pid]
        q = api(host, "/queue")
        running = [r for r in q.get("queue_running", []) if r[1] == pid]
        pending = [r for r in q.get("queue_pending", []) if r[1] == pid]
        state = "running" if running else ("queued" if pending else "…")
        if state != last:
            print(f"  [{time.time() - t0:6.1f}s] {state}")
            last = state
        elif not running and not pending:
            # left the queue without a history entry: it failed to start
            time.sleep(poll)
            hist = api(host, f"/history/{pid}")
            if pid not in hist:
                return time.time() - t0, None
            return time.time() - t0, hist[pid]
        time.sleep(poll)


def report(elapsed, entry, chunks):
    print(f"\n  wall clock: {elapsed:.1f}s", end="")
    if chunks:
        print(f"  ({elapsed / chunks:.1f}s per chunk over {chunks})")
    else:
        print()
    if entry is None:
        print("  NO HISTORY ENTRY — it never ran. Check the ComfyUI console.")
        return 1
    status = (entry.get("status") or {})
    if status.get("status_str") and status["status_str"] != "success":
        print(f"  STATUS: {status['status_str']}")
        for m in (status.get("messages") or [])[-6:]:
            print(f"    {m}")
        return 1
    vids = []
    for out in (entry.get("outputs") or {}).values():
        for key in ("images", "gifs", "video", "videos", "audio"):
            for f in out.get(key, []) or []:
                if isinstance(f, dict) and f.get("filename"):
                    vids.append(f"{f.get('subfolder','')}/{f['filename']}".lstrip("/"))
    if vids:
        print("  OUTPUT:")
        for v in dict.fromkeys(vids):
            print(f"    {v}")
    else:
        print("  ran, but declared no output file — check the SaveVideo node")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", help="an API-format workflow JSON")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--submit", action="store_true",
                    help="actually queue it; without this, validate only")
    ap.add_argument("--chunks", type=int, default=0,
                    help="shorten the take to N chunks by lowering chunk_count "
                         "(0 = leave the graph alone). 2 is the cheapest proof "
                         "of the chunked path.")
    ap.add_argument("--watch", action="store_true", help="wait for it to finish")
    a = ap.parse_args()

    graph = load_graph(a.graph)
    kinds = {}
    for n in graph.values():
        kinds[n["class_type"]] = kinds.get(n["class_type"], 0) + 1
    print(f"{pathlib.Path(a.graph).name}: {len(graph)} nodes")
    chunked = [k for k in kinds if k.startswith("H3Chunk")]
    print(f"  chunking nodes: {', '.join(sorted(chunked)) or 'NONE — not a chunked graph'}")

    try:
        problems, notes = validate(a.host, graph)
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"cannot reach ComfyUI at {a.host}: {e}")
    if notes:
        print(f"  {len(notes)} frontend extra(s), harmless:")
        for n in notes[:4]:
            print(f"    {n}")
    if problems:
        print(f"\n{len(problems)} problem(s) — not submitting:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("  validates against the live server")

    if a.chunks:
        for line in chunk_limit(graph, a.chunks):
            print(f"  {line}")
    p = plan_of(a.host, graph)
    if p and p.get("ok"):
        print(f"  plan: {p['chunk_count']} x {p['chunk_frames']} carrying "
              f"{p['context']} -> {p['total_frames']}f = {p['total_seconds']}s, "
              f"{p['beats_required']} beat(s)")
    if not a.submit:
        print("\n(validate only; pass --submit to queue it)")
        return 0

    pid = submit(a.host, graph)
    print(f"\nqueued {pid}")
    if not a.watch:
        print("(pass --watch to wait; or poll /history/%s)" % pid)
        return 0
    elapsed, entry = watch(a.host, pid)
    return report(elapsed, entry, a.chunks or (p or {}).get("chunk_count"))


if __name__ == "__main__":
    sys.exit(main())
