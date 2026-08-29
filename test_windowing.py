"""Window scheduling arithmetic. No torch, no ComfyUI.

`window_schedule` is a REPLICA of comfy.context_windows.create_windows_static_
standard, kept in this pack so a plan can be read without importing comfy (which
initialises CUDA). A replica that drifts from the original is worse than no
replica, so the cases here are the ones that pin its behaviour: the clamp, the
stride, and the frame grid.

    python3 test_windowing.py
"""

import sys
import types

sys.modules.setdefault("torch", types.ModuleType("torch"))

from timing import video_latent_t                        # noqa: E402
from windowing import frames_for_latent, window_schedule  # noqa: E402

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")


def ok(label, cond):
    if not cond:
        fails.append(label)


# --- the latent grid round-trips ------------------------------------------- #
# legal runs are 17n+5 and legal latent counts are 5n+2; the two must agree in
# both directions or a window lands between frames
for f in (5, 22, 39, 56, 90, 141, 192, 243, 294, 345, 396, 447):
    check(f"round trip {f}", frames_for_latent(video_latent_t(f)), f)
check("latent 2 is the floor", frames_for_latent(2), 5)
check("192 frames", video_latent_t(192), 57)
check("345 frames", video_latent_t(345), 102)

# --- the schedule ---------------------------------------------------------- #
# 345 frames at window 141 / overlap 39 divides exactly: three windows, each
# starting a full stride after the last, no clamp
sched = window_schedule(102, 42, 12)
check("345/141/39 is three clean windows", sched, [(0, 42), (30, 72), (60, 102)])
ok("and every overlap is the one asked for",
   all(sched[i - 1][1] - sched[i][0] == 12 for i in range(1, len(sched))))

# 192 frames at the same settings does NOT divide. The last window is pulled
# BACK to fit, so it shares 27 latent frames rather than 12 -- silently a
# gentler run than the settings say, and not comparable to one that divides.
sched = window_schedule(57, 42, 12)
check("192/141/39 clamps to two", sched, [(0, 42), (15, 57)])
check("and the real overlap is bigger than asked", sched[0][1] - sched[1][0], 27)

# a clip shorter than one window is a single window, not a negative start
check("short clip", window_schedule(20, 42, 12), [(0, 42)])
ok("never starts before zero",
   all(a >= 0 for a, _ in window_schedule(30, 42, 12)))

# the 90-frame window the earlier measurements used
check("192/90/39", window_schedule(57, 27, 12), [(0, 27), (15, 42), (30, 57)])

# --- the audio grid -------------------------------------------------------- #
# every window must start on an exact 40 Hz tick, so the PIXEL stride has to
# divide by 3. At a 90 window that leaves exactly one legal overlap.
def stride_frames(wf, of):
    d = video_latent_t(wf) - video_latent_t(of)
    return frames_for_latent(d + 2) - 5


check("90/39 stride", stride_frames(90, 39), 51)
check("90/22 stride", stride_frames(90, 22), 68)
check("90/5 stride", stride_frames(90, 5), 85)
check("141/39 stride", stride_frames(141, 39), 102)
legal = [o for o in range(5, 90, 17) if stride_frames(90, o) % 3 == 0]
check("39 is the only both-clocks overlap at a 90 window", legal, [39])
ok("102 divides by 3, so 141/39 is on both clocks",
   stride_frames(141, 39) % 3 == 0)

# --- the frame phase ------------------------------------------------------- #
# FRAME_PER_TOKEN is (1,4,4,4,4), so a stride that is not a multiple of 5 latent
# frames puts every window on a different intra-window frame grid
for wf, of in ((90, 39), (90, 22), (141, 39), (192, 39)):
    d = video_latent_t(wf) - video_latent_t(of)
    ok(f"{wf}/{of} latent stride is a multiple of 5", d % 5 == 0)

if fails:
    print("FAIL")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("windowing: all checks pass")
