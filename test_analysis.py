"""Offline tests for H3DriftCheck — no ComfyUI, no torch.

    python3 test_analysis.py

The stub here goes further than the ones in test_chunkrun.py and
test_latentpin.py, which fake torch just well enough for the SHAPES to flow. The
thing under test is a measurement, so a shape-only fake would prove the table
prints and prove nothing about the figure in it. numpy is already a hard
dependency of every host this runs on, and it agrees with torch on the handful of
operations analysis.py uses, so `torch` here is a thin alias over it and the
arithmetic being checked is the real arithmetic.
"""

import sys
import types

import numpy as np


class A(np.ndarray):
    """A numpy array that answers to the two torch method names in use.

    Slices, boolean indexing and arithmetic all preserve the subclass, so this
    only has to add what ndarray genuinely lacks.
    """

    def permute(self, *dims):
        return self.transpose(*dims)


def arr(x):
    return np.asarray(x, dtype=np.float32).view(A)


def _interpolate(x, scale_factor=1.0, mode=None, align_corners=None):
    """Nearest-neighbour stand-in for F.interpolate on [N,C,H,W].

    Deliberately not bilinear: the point of this stub is that the code paths
    around it run, and a resampler that filters would put the stub's own
    behaviour into the measured numbers. The drift assertions below therefore
    run at scale 1.0, where no resampling happens at all, and scale 0.5 is
    checked only for the shape it produces.
    """
    step = max(1, int(round(1.0 / float(scale_factor))))
    return x[:, :, ::step, ::step]


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.abs = lambda x: np.abs(x).view(A)
    torch.amax = lambda x, dim: np.amax(x, dim).view(A)
    torch.amin = lambda x, dim: np.amin(x, dim).view(A)
    torch.cat = lambda xs, dim=0: np.concatenate(xs, axis=dim).view(A)
    torch.tensor = lambda v, device=None, dtype=None: arr(v)
    nn = types.ModuleType("torch.nn")
    fn = types.ModuleType("torch.nn.functional")
    fn.interpolate = _interpolate
    nn.functional = fn
    torch.nn = nn
    torch.__path__ = []
    nn.__path__ = []
    for name, mod in (("torch", torch), ("torch.nn", nn),
                      ("torch.nn.functional", fn)):
        sys.modules.setdefault(name, mod)


_install_stubs()

import analysis  # noqa: E402
import chunkplan  # noqa: E402


FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r}, wanted {want!r}")
        print(f"  FAIL {name}: got {got!r}, wanted {want!r}")
    else:
        print(f"  ok   {name}")


def real_plan(total=720):
    """The plan the drift was measured on: 4 chunks, 753 frames kept."""
    chunks, info = chunkplan.plan(total, chunk_frames=243, context=39,
                                  grow_tail=True, generated_audio=True)
    return {"chunks": chunks, "info": info, "total_frames": total}


def clip(spans, h=64, w=64, amplitudes=(), skin=True):
    """A clip whose skin-coloured torso band carries a known micro-texture.

    Base colour satisfies the mask (R>G>B, saturation inside 0.18–0.62, max
    channel inside 0.25–0.98); a per-pixel checkerboard of the given amplitude
    supplies local contrast that the 3×3 residual has to recover, so a rising
    amplitude per span must come back as a rising detail column.
    """
    n = spans[-1][1]
    base = np.zeros((n, h, w, 3), dtype=np.float32)
    if skin:
        base[..., 0], base[..., 1], base[..., 2] = 0.62, 0.46, 0.36
    else:
        base[..., :] = 0.5           # grey: R>G>B fails, so no skin pixels
    board = np.indices((h, w)).sum(0) % 2 * 2.0 - 1.0
    for (a, b), amp in zip(spans, amplitudes):
        base[a:b] += (board * amp)[None, :, :, None]
    return arr(np.clip(base, 0.0, 1.0))


def _rows(info):
    """The table rows: indented, starting with a chunk number."""
    out = []
    for line in info.splitlines():
        head = line.strip().split(" ")[0]
        if line.startswith("  ") and head.isdigit():
            out.append(line)
    return out


def test_joined_positions():
    print("chunk boundaries in the JOINED clip, not source indices")
    plan = real_plan()
    kept = [int(c["end"]) - int(c["keep_from"]) for c in plan["chunks"]]
    check("the plan is the measured one: 4 chunks", len(plan["chunks"]), 4)
    check("kept lengths sum to 753", sum(kept), 753)

    segs = analysis.segments(plan, 753)
    check("untrimmed spans", [(a, b) for a, b, _ in segs],
          [(0, 243), (243, 447), (447, 651), (651, 753)])

    # source `start` overlaps by the context (204, 408, 612) and would put every
    # chunk after the first in the wrong place — this is the whole trap
    starts = [int(c["start"]) for c in plan["chunks"]]
    check("source starts differ from joined starts", starts,
          [0, 204, 408, 612])

    # H3 Chunk Close trims the grown tail back to the 720 that was asked for
    trimmed = analysis.segments(plan, 720)
    check("trimmed clip clamps the last span", [(a, b) for a, b, _ in trimmed],
          [(0, 243), (243, 447), (447, 651), (651, 720)])

    # and a clip shorter than the plan drops the chunks that fall off entirely
    short = analysis.segments(plan, 300)
    check("a short clip drops chunks past the end",
          [(a, b) for a, b, _ in short], [(0, 243), (243, 300)])

    check("no plan means no segments", analysis.segments(None, 720), [])
    check("an empty plan means no segments", analysis.segments({}, 720), [])


def test_per_chunk_aggregation():
    print("per-chunk aggregation, and the drift it is built to see")
    plan = real_plan()
    spans = [(a, b) for a, b, _ in analysis.segments(plan, 720)]
    images = clip(spans, amplitudes=(0.02, 0.03, 0.04, 0.05))

    sheet, info = analysis.H3DriftCheck().go(images, plan=plan,
                                             scale=1.0)["result"]
    rows = _rows(info)
    check("one row per chunk", len(rows), 4)
    check("header names the segment count", "4 segment(s)" in info, True)
    check("row 2 spans the joined clip, not the source", "243-446" in rows[1],
          True)
    check("row 4 stops where the trim did", "651-719" in rows[3], True)

    detail = [float(r.split()[3]) for r in rows]
    check("detail rises with the planted micro-texture",
          detail == sorted(detail) and detail[0] < detail[-1], True)
    check("chunk 1 is the baseline, so no comparison on its row",
          rows[0].split()[4], "—")
    check("later rows carry a positive vs-base figure",
          all(r.split()[4].startswith("+") for r in rows[1:]), True)

    # the failed proxies stay in the table because they are the ones that do NOT
    # move; a flat checkerboard changes neither the level nor the colour
    luma = [float(r.split()[6]) for r in rows]
    sat = [float(r.split()[7]) for r in rows]
    check("luma is flat across the chunks", max(luma) - min(luma) < 0.01, True)
    check("saturation is flat across the chunks", max(sat) - min(sat) < 0.01,
          True)

    # one representative frame per chunk, side by side, in the analysed region
    h = int(64 * 0.72) - int(64 * 0.30)
    w = int(64 * 0.85) - int(64 * 0.15)
    check("contact sheet is 4 band frames wide", tuple(sheet.shape),
          (1, h, w * 4, 3))

    _, whole = analysis.H3DriftCheck().go(images, plan=plan, scale=1.0,
                                          region="whole frame")["result"]
    check("whole frame is a different measurement", whole != info, True)

    half, _ = analysis.H3DriftCheck().go(images, plan=plan,
                                         scale=0.5)["result"]
    check("scale halves the sheet", tuple(half.shape),
          (1, (h + 1) // 2, ((w + 1) // 2) * 4, 3))


def test_no_plan_fallback():
    print("without a plan, one segment and no attribution")
    images = clip([(0, 120)], amplitudes=(0.03,))
    sheet, info = analysis.H3DriftCheck().go(images, scale=1.0)["result"]
    check("one row", len(_rows(info)), 1)
    check("the row covers the whole clip", "0-119" in _rows(info)[0], True)
    check("header says one segment", "1 segment(s)" in info, True)
    check("it says the numbers cannot be attributed",
          "NO PLAN WIRED" in info, True)
    check("one frame in the sheet", int(sheet.shape[2]),
          int(64 * 0.85) - int(64 * 0.15))

    # a single-chunk plan IS attributed, so the warning must not appear
    one = {"chunks": [{"start": 0, "end": 120, "keep_from": 0}]}
    _, attributed = analysis.H3DriftCheck().go(images, plan=one,
                                               scale=1.0)["result"]
    check("a one-chunk plan is not the fallback",
          "NO PLAN WIRED" in attributed, False)


def test_skips_rather_than_inventing():
    print("a chunk with no skin reports nothing rather than a number")
    images = clip([(0, 60)], amplitudes=(0.03,), skin=False)
    _, info = analysis.H3DriftCheck().go(images, scale=1.0)["result"]
    check("the row says why it is empty", "too few skin pixels" in info, True)
    check("no baseline is claimed", "baseline:" in info, False)
    check("no detail figure is printed", "0.0" in _rows(info)[0], False)


def test_skipped_first_chunk_does_not_become_the_baseline():
    print("a skipped chunk 1 must not be compared against")
    plan = {"chunks": [{"start": 0, "end": 60, "keep_from": 0},
                       {"start": 0, "end": 180, "keep_from": 60}]}
    skinless = clip([(0, 60)], amplitudes=(0.03,), skin=False)
    warm = clip([(0, 120)], amplitudes=(0.03,))
    images = arr(np.concatenate([skinless, warm]))
    _, info = analysis.H3DriftCheck().go(images, plan=plan, scale=1.0)["result"]
    rows = _rows(info)
    check("two rows", len(rows), 2)
    check("chunk 1 reports nothing", "too few skin pixels" in rows[0], True)
    check("chunk 2 becomes the baseline, so no comparison on its row",
          rows[1].split()[4], "—")
    check("the column is not called vs c1", "vs c1" in info, False)

    # a clip longer than the plan covers must say so rather than quietly dropping
    longer = arr(np.concatenate([images, warm[:30]]))
    _, note = analysis.H3DriftCheck().go(longer, plan=plan, scale=1.0)["result"]
    check("uncovered tail frames are named",
          "30 frame(s) past the plan's last chunk" in note, True)


def test_reports_only():
    print("no threshold, no verdict")
    plan = real_plan()
    spans = [(a, b) for a, b, _ in analysis.segments(plan, 720)]
    images = clip(spans, amplitudes=(0.02, 0.03, 0.04, 0.05))
    _, info = analysis.H3DriftCheck().go(images, plan=plan,
                                         scale=1.0)["result"]
    # "failed" and "threshold" DO appear, in the sentence saying there is no
    # threshold because every proxy has failed — so match verdict words that
    # cannot turn up innocently rather than banning the disclaimer's vocabulary
    for word in ("good", "bad", "acceptable", "too high", "too low", "expected",
                 "recommend", "you should", "warning", "problem", "ok "):
        check(f"no {word.strip()!r} verdict", word in info.lower(), False)
    check("the disclaimer is there", "no threshold and no verdict" in info, True)
    check("units are stated", "dimensionless" in info, True)
    check("the baseline is stated", "baseline:" in info, True)


def test_ui_shape():
    print("the return shape other long-form nodes use")
    images = clip([(0, 60)], amplitudes=(0.03,))
    out = analysis.H3DriftCheck().go(images, scale=1.0)
    check("ui carries the text under h3char",
          out["ui"]["h3char"][0] == out["result"][1], True)
    check("two outputs", len(out["result"]), 2)
    check("slot order is (frames, info)", analysis.H3DriftCheck.RETURN_NAMES,
          ("frames", "info"))
    check("types match the names", analysis.H3DriftCheck.RETURN_TYPES,
          ("IMAGE", "STRING"))
    # widgets and links are positional; anything new belongs after `scale`
    check("optional widgets stay in order",
          list(analysis.H3DriftCheck.INPUT_TYPES()["optional"]),
          ["plan", "region", "scale"])
    check("required inputs stay in order",
          list(analysis.H3DriftCheck.INPUT_TYPES()["required"]), ["images"])


def main():
    for fn in (test_joined_positions, test_per_chunk_aggregation,
               test_no_plan_fallback, test_skips_rather_than_inventing,
               test_skipped_first_chunk_does_not_become_the_baseline,
               test_reports_only, test_ui_shape):
        fn()
    print()
    if FAILED:
        print(f"{len(FAILED)} failure(s)")
        return 1
    print("all drift-check checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
