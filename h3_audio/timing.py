"""H3's two clocks, and the frame counts where they agree.

H3 runs video at 24 fps and audio on a 40 Hz latent grid, and a clip's length has
to be legal on BOTH. The video VAE only accepts runs of 17n+5 frames, which is the
constraint everything in this pack already respected. The audio constraint is
newer to us and was easy to miss, because nothing errors when you break it:

    audio_t = round(frames / 24 * 40)

`round` is doing real work there. A run only lands exactly on the audio grid when
frames * 40 is divisible by 24 — otherwise the audio latent is rounded to the
nearest step and the clip's audio is a fraction of a step longer or shorter than
its video. One clip, nobody notices. A CHAIN of clips accumulates it, and the
offsets handed to H3 Audio Lock drift away from where the picture actually is.

Both constraints together: frames = 17k + 5 AND frames % 3 == 0, which is every
third video run — 39, 90, 141, 192, 243, 294, 345, 396, spaced 51 apart.

    39 frames =  1.625 s =  65 audio steps
    90 frames =  3.750 s = 150
   141 frames =  5.875 s = 235
   192 frames =  8.000 s = 320
   243 frames = 10.125 s = 405
   294 frames = 12.250 s = 490
   345 frames = 14.375 s = 575

Our long-form work used 362 frames throughout, which is a legal VIDEO run and is
NOT on the audio grid: 362 * 40 / 24 = 603.33, so every link carries a third of an
audio step of error. 345 is the aligned run nearest to it and costs 0.71 s.

Credit: the alignment rule is from seitanism/ComfyUI-H3-Motion-Context-MultiRef
(`h3_timing.py`), a fork of NikoDemon80's H3 Motion Context — the same pack whose
audio-context coordinate trick is already credited in the engine's packed_kf.py.
"""

FPS = 24
AUDIO_LATENT_HZ = 40


def align_frames(seconds):
    """Seconds -> the next legal video run (17n+5). Ignores the audio grid."""
    n = max(5, round(float(seconds) * FPS))
    while n % 17 != 5:
        n += 1
    return n


def video_runs_through(limit):
    """Every legal video-VAE run up to `limit`: 5, 22, 39, 56, ..."""
    return list(range(5, int(limit) + 1, 17))


def is_av_aligned(frames):
    """True when this run's end lands exactly on the 40 Hz audio latent grid."""
    return (int(frames) * AUDIO_LATENT_HZ) % FPS == 0


def av_aligned_runs_through(limit):
    """Legal video runs that ALSO land exactly on the audio clock."""
    return [n for n in video_runs_through(limit) if is_av_aligned(n)]


def audio_t(frames):
    """The audio latent length H3 will actually allocate for this run."""
    return round(int(frames) / FPS * AUDIO_LATENT_HZ)


def av_error_steps(frames):
    """How far off the audio grid this run is, in audio latent steps (0 to 0.5)."""
    exact = int(frames) / FPS * AUDIO_LATENT_HZ
    return abs(exact - round(exact))


def snap_av_aligned(frames, direction="nearest"):
    """Move a legal run to the nearest AV-aligned one.

    direction: "nearest" | "down" | "up". Returns `frames` unchanged if it is
    already aligned. Never returns less than 39, the smallest aligned run.
    """
    n = max(5, int(frames))
    if is_av_aligned(n):
        return n
    down = n
    while down >= 39 and not is_av_aligned(down):
        down -= 17
    up = n
    while not is_av_aligned(up):
        up += 17
    if direction == "down":
        return down if down >= 39 else up
    if direction == "up":
        return up
    if down < 39:
        return up
    return down if (n - down) <= (up - n) else up


def describe(frames):
    """One line for a node's info output."""
    n = int(frames)
    line = f"{n} frames = {n / FPS:.3f}s"
    if is_av_aligned(n):
        return line + f" = {audio_t(n)} audio steps (AV-aligned)"
    return (line + f", audio {n / FPS * AUDIO_LATENT_HZ:.3f} -> {audio_t(n)} steps "
            f"(off the audio grid by {av_error_steps(n):.2f} of a step; "
            f"nearest aligned run is {snap_av_aligned(n)})")
