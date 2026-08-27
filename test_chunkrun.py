"""Offline tests for the chunk runner — no ComfyUI, no torch.

The runner is the one part of the pack that rewrites a GRAPH, so a mistake here
is silent: links land on the wrong slot and every chunk renders something
plausible but wrong. These tests stand in fake `comfy_execution.graph_utils` and
`torch` modules so the slicing arithmetic and the expansion can both run here.

    python3 test_chunkrun.py
"""

import sys
import types


class Fake:
    """Enough of a tensor to be sliced and measured."""

    def __init__(self, n, tag="img"):
        self.n = n
        self.tag = tag

    @property
    def shape(self):
        return (self.n, 64, 64, 3)

    @property
    def device(self):
        return "cpu"

    def __getitem__(self, key):
        start, stop, _ = key.indices(self.n)
        return Fake(max(0, stop - start), f"{self.tag}[{start}:{stop}]")

    def __repr__(self):
        return f"Fake({self.tag}, {self.n})"


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.zeros = lambda shape, device=None: Fake(shape[0], "zeros")
    sys.modules.setdefault("torch", torch)

    class Node:
        def __init__(self, graph, class_type, nid, kw):
            self.graph, self.class_type, self.id = graph, class_type, nid
            self.inputs = dict(kw)

        def set_input(self, key, val):
            self.inputs[key] = val

        def out(self, index):
            return [self.id, index]

    class GraphBuilder:
        def __init__(self):
            self.nodes = {}
            self._n = 0

        def node(self, class_type, id=None, **kw):
            if id is None:
                self._n += 1
                id = f"auto{self._n}"
            n = Node(self, class_type, id, kw)
            self.nodes[id] = n
            return n

        def finalize(self):
            return {nid: {"class_type": n.class_type, "inputs": n.inputs}
                    for nid, n in self.nodes.items()}

    mod = types.ModuleType("comfy_execution.graph_utils")
    mod.GraphBuilder = GraphBuilder
    mod.is_link = lambda v: isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)
    pkg = types.ModuleType("comfy_execution")
    pkg.graph_utils = mod
    sys.modules.setdefault("comfy_execution", pkg)
    sys.modules.setdefault("comfy_execution.graph_utils", mod)


_install_stubs()

import chunkrun  # noqa: E402
import chunkplan  # noqa: E402


def make_plan(total, chunk_frames=90):
    """The H3_CHUNK_PLAN shape the nodes pass around."""
    chunks, info = chunkplan.plan(total, chunk_frames=chunk_frames)
    return {"chunks": chunks, "info": info, "total_frames": total}


class DynPrompt:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node(self, nid):
        return self.nodes[nid]

    def all_node_ids(self):
        return list(self.nodes)


FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r}, wanted {want!r}")
        print(f"  FAIL {name}: got {got!r}, wanted {want!r}")
    else:
        print(f"  ok   {name}")


def test_v2v_slices():
    print("V2V — every stream cut on the same boundary")
    p = make_plan(180, chunk_frames=90)
    imgs, msk, aud, length, idx, flow, info, count, kf, ctx, xtra = (
        chunkrun.slice_chunk(p, 1, Fake(180), mask=Fake(180, "mask"),
                             extra_images=Fake(180, "depth")))
    c = p["chunks"][1]
    check("images length", imgs.n, c["end"] - c["start"])
    check("mask length", msk.n, c["end"] - c["start"])
    check("extra length", xtra.n, c["end"] - c["start"])
    check("extra cut where images were", xtra.tag,
          f"depth[{c['start']}:{c['end']}]")
    check("reported length", length, c["end"] - c["start"])
    check("chunk_count", count, len(p["chunks"]))
    check("no keyframe in V2V", kf, None)


def test_extra_optional():
    print("V2V — extra unwired stays None, nothing else shifts")
    p = make_plan(180, chunk_frames=90)
    out = chunkrun.slice_chunk(p, 0, Fake(180))
    check("output arity", len(out), 11)
    check("extra is None", out[10], None)
    check("mask synthesised", out[1].tag, "zeros")


def test_fresh_generation():
    print("fresh generation — keyframe is the last frame, context is the tail")
    p = make_plan(180, chunk_frames=90)
    out = chunkrun.slice_chunk(p, 1, None, prev_images=Fake(90, "prev"),
                               context_frames=39)
    check("no images", out[0], None)
    check("length from the plan", out[3], p["chunks"][1]["run"])
    check("keyframe is one frame", out[8].n, 1)
    check("keyframe is the LAST frame", out[8].tag, "prev[89:90]")
    check("context is the tail", out[9].tag, "prev[51:90]")
    check("no extra without a source", out[10], None)


def test_context_clamped():
    print("context longer than the previous chunk clamps, never wraps")
    out = chunkrun.slice_chunk({"chunks": [{"start": 0, "end": 0, "run": 39,
                                            "both_clocks": True,
                                            "seed_mask": False}]},
                               0, None, prev_images=Fake(10, "prev"),
                               context_frames=39)
    check("context clamped to what exists", out[9].n, 10)


def _swap_graph():
    """Open -> conditioning -> sampler -> decode -> Close, with two externals."""
    return DynPrompt({
        "load": {"class_type": "VHS_LoadVideo", "inputs": {}},
        "vae": {"class_type": "VAELoader", "inputs": {}},
        "depth": {"class_type": "DepthAnything_V2", "inputs": {"images": ["load", 0]}},
        "plan": {"class_type": "H3ChunkPlan", "inputs": {"source_images": ["load", 0]}},
        "open": {"class_type": "H3ChunkOpen",
                 "inputs": {"plan": ["plan", 0], "source_images": ["load", 0],
                            "extra_images": ["depth", 0], "context_frames": 39}},
        "cond": {"class_type": "MiniMaxH3ReferenceToVideo",
                 "inputs": {"vae": ["vae", 0], "length": ["open", 3],
                            "ref_video_1": ["open", 10], "ref_image_1": ["ref", 0]}},
        "samp": {"class_type": "SamplerCustomAdvanced",
                 "inputs": {"latent_image": ["cond", 1]}},
        "dec": {"class_type": "VAEDecode",
                "inputs": {"samples": ["samp", 0], "vae": ["vae", 0]}},
        "ref": {"class_type": "LoadImage", "inputs": {}},
        "close": {"class_type": "H3ChunkClose",
                  "inputs": {"flow": ["open", 5], "images": ["dec", 0]}},
    })


def test_body_capture():
    print("Close clones the body and only the body")
    dp = _swap_graph()
    p = make_plan(180, chunk_frames=90)
    res = chunkrun.H3ChunkClose().go({"plan": p, "index": 0}, Fake(90),
                                     dynprompt=dp, unique_id="close")
    graph = res["expand"]
    kinds = sorted({n["class_type"] for n in graph.values()})
    check("cloned classes", kinds,
          ["H3ChunkSlice", "ImageBatch", "MiniMaxH3ReferenceToVideo",
           "SamplerCustomAdvanced", "VAEDecode"])
    check("one slicer per chunk",
          sum(1 for n in graph.values() if n["class_type"] == "H3ChunkSlice"),
          len(p["chunks"]))
    check("one join per seam",
          sum(1 for n in graph.values() if n["class_type"] == "ImageBatch"),
          len(p["chunks"]) - 1)
    check("VAE loader NOT cloned",
          sum(1 for n in graph.values() if n["class_type"] == "VAELoader"), 0)
    check("depth NOT cloned — it is computed once on the whole clip",
          sum(1 for n in graph.values() if n["class_type"] == "DepthAnything_V2"), 0)


def test_links_remapped():
    print("links from Open land on the slicer, externals keep pointing out")
    dp = _swap_graph()
    p = make_plan(180, chunk_frames=90)
    graph = chunkrun.H3ChunkClose().go({"plan": p, "index": 0}, Fake(90),
                                       dynprompt=dp, unique_id="close")["expand"]
    cond1 = graph["c1_cond"]["inputs"]
    check("length from chunk 1's slicer", cond1["length"], ["slice1", 3])
    check("ref_video keeps slot 10", cond1["ref_video_1"], ["slice1", 10])
    check("shared VAE still external", graph["c1_dec"]["inputs"]["vae"], ["vae", 0])
    check("shared ref image still external", cond1["ref_image_1"], ["ref", 0])
    s1 = graph["slice1"]["inputs"]
    check("slicer index", s1["chunk_index"], 1)
    check("slicer fed by the shared loader", s1["source_images"], ["load", 0])
    check("slicer fed by the shared depth", s1["extra_images"], ["depth", 0])
    check("V2V still threads prev_images", s1["prev_images"], ["c0_dec", 0])
    check("chunk 0 has no prev", "prev_images" in graph["slice0"]["inputs"], False)


def test_overlap_trimmed_at_the_join():
    print("an extended tail is trimmed back so no frame appears twice")
    p = make_plan(400, chunk_frames=90)      # 4x90 then a 40f tail -> 56f run
    tail = p["chunks"][-1]
    check("the tail reaches back for a legal run",
          (tail["start"], tail["end"], tail["run"]), (344, 400, 56))
    check("and marks where its new content starts", tail["keep_from"], 360)

    graph = chunkrun.H3ChunkClose().go({"plan": p, "index": 0}, Fake(90),
                                       dynprompt=_swap_graph(),
                                       unique_id="close")["expand"]
    trims = {nid: n for nid, n in graph.items()
             if n["class_type"] == "ImageFromBatch"}
    check("one trim, on the one overlapping chunk", len(trims), 1)
    t = list(trims.values())[0]["inputs"]
    check("drops the regenerated head", t["batch_index"], 16)
    check("keeps the new frames", t["length"], 40)
    check("trims the tail chunk's own output", t["image"], ["c4_dec", 0])
    check("the join reads the TRIMMED tail, not the raw one",
          any(list(trims)[0] in (v[0] if isinstance(v, list) else None
                                 for v in n["inputs"].values())
              for n in graph.values() if n["class_type"] == "ImageBatch"), True)

    # nothing to trim when the clip divides exactly
    even = make_plan(360, chunk_frames=90)
    g2 = chunkrun.H3ChunkClose().go({"plan": even, "index": 0}, Fake(90),
                                    dynprompt=_swap_graph(),
                                    unique_id="close")["expand"]
    check("no trim when every chunk is already legal",
          sum(1 for n in g2.values() if n["class_type"] == "ImageFromBatch"), 0)


def test_refuses_bad_wiring():
    print("Close refuses rather than silently rendering one chunk")
    p = make_plan(180, chunk_frames=90)
    dp = DynPrompt({"dec": {"class_type": "VAEDecode", "inputs": {}},
                    "close": {"class_type": "H3ChunkClose",
                              "inputs": {"images": ["dec", 0]}}})
    _, info = chunkrun.H3ChunkClose().go({"plan": p}, Fake(90),
                                         dynprompt=dp, unique_id="close")
    check("no Open upstream is reported", info.startswith("No H3 Chunk Open"), True)

    one = make_plan(90, chunk_frames=90)
    _, info = chunkrun.H3ChunkClose().go({"plan": one}, Fake(90),
                                         dynprompt=_swap_graph(),
                                         unique_id="close")
    check("single chunk passes through", "one chunk" in info, True)


def main():
    for fn in (test_v2v_slices, test_extra_optional, test_fresh_generation,
               test_context_clamped, test_body_capture, test_links_remapped,
               test_overlap_trimmed_at_the_join, test_refuses_bad_wiring):
        fn()
    print()
    if FAILED:
        print(f"{len(FAILED)} failure(s)")
        return 1
    print("all chunk-runner checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
