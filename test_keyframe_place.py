"""Where a keyframe actually lands — the decision the timeline node exists to make.

THE GRID IS NOT COSMETIC. `FRAME_PER_TOKEN` is (1,4,4,4,4), so latent step k
covers ONE pixel frame when k % 5 == 0 and four otherwise. The clean anchors are
the multiples of 17; land between them and the keyframe is smeared across the
four frames its step covers. Asking for 5.0s gives frame 120 when the clean slot
is 119, and that is a plausible part of why a 5s anchor did not visibly pull on
2026-09-09.

The LAST frame is the exception: it is not on the 17 grid, and it is a proven
anchor anyway because core places `last_frame` at frame_count - 1. Snapping it
back to 136 would move a trodden position onto an untrodden one.

`_place` is pulled out of the class and exec'd because video.py imports ComfyUI
at module level and there is no torch in WSL.

    python3 test_keyframe_place.py
"""
import pathlib
import sys

src = pathlib.Path(__file__).parent.joinpath("video.py").read_text()
start = src.index("    @staticmethod\n    def _place")
end = src.index("    def go(self, conditioning, vae, length, **kw):")
ns = {}
exec(compile("class T:\n" + src[start:end], "<place>", "exec"), ns)
place = ns["T"]._place

fails = []


def check(label, got, want):
    if got != want:
        fails.append(label)
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


notes = []
print("a 141-frame clip: clean slots are 0, 17, 34 ... 136, plus the last frame")
check("5.0s is frame 120 and snaps to 119", place(120, 141, True, notes, 2, "5s"), 119)
check("frame 0 is already a slot", place(0, 141, True, notes, 1, "f"), 0)
check("frame 119 is already a slot", place(119, 141, True, notes, 2, "f"), 119)
check("60 goes up to 68, not down to 51", place(60, 141, True, notes, 3, "f"), 68)
check("55 goes down to 51", place(55, 141, True, notes, 3, "f"), 51)
check("138 goes to 136 — the slot above it is past the clip",
      place(138, 141, True, notes, 4, "f"), 136)

print("the last frame is its own anchor and must NOT be snapped back")
check("past the end clamps to 140, not 136", place(9999, 141, True, notes, 4, "f"), 140)
check("and the note says it is the last frame",
      any("LAST frame" in n for n in notes), True)

print("snapping is optional, clamping is not")
check("snap off keeps 120 exactly", place(120, 141, False, notes, 2, "f"), 120)
check("negative clamps to 0", place(-5, 141, True, notes, 1, "f"), 0)
check("a shorter clip clamps to ITS last frame", place(120, 90, True, notes, 1, "5s"), 89)

print("every placement is reported, snapped or not")
n2 = []
place(120, 141, True, n2, 2, "5s")
check("a moved keyframe says where it went", "-> frame 119" in n2[0], True)
n3 = []
place(120, 141, False, n3, 2, "5s")
check("an unmoved off-grid keyframe says it is smeared",
      "NOT on the 17-frame grid" in n3[0], True)

print()
print(f"{len(fails)} failure(s)" if fails else "keyframe placement: all checks pass")
sys.exit(1 if fails else 0)
