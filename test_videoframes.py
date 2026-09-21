"""The frame arithmetic — where every silent bug in a video loader lives.

WHAT THIS IS DEFENDING

  Rate conversion, frame selection and trimming compose, and the composition is
  invisible from outside the node. Get it wrong and nothing errors: a clip comes
  out the wrong length, or starts a few frames off, and the mistake surfaces
  somewhere else entirely -- as a model complaining about a frame count, or as a
  cut landing in the wrong place.

  The specific trap this file exists for: "frame 192" means three different
  things depending on which domain you are in, and a timeline drawn in the wrong
  one lies to you the moment force_rate is not the source's own rate.

    python3 test_videoframes.py
"""
import sys

import videoframes as f

fails = []


def check(label, got, want):
    if got != want:
        fails.append(label)
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


def close(label, got, want, tol=1e-6):
    check(label, abs(got - want) < tol, True)


print("rate conversion goes through DURATION, not through the frame count")
# 300 frames at 30 fps is 10s; 10s at 24 fps is 240 frames.
check("30 -> 24 fps shortens the count", f.rate_length(300, 30, 24), 240)
check("no force_rate leaves it alone", f.rate_length(300, 30, 0), 300)
check("force_rate equal to source is a no-op", f.rate_length(300, 30, 30), 300)
check("24 -> 30 fps lengthens it", f.rate_length(240, 24, 30), 300)
# THE CASE THAT MAKES THE METHOD MATTER. Scaling the count by 24/30 agrees with
# converting through duration at clean rates and stops agreeing at 29.97 --
# which is most real footage.
check("29.97 is handled as a duration, not a ratio",
      f.rate_length(300, 29.97, 24), 240)
check("a zero-length source stays zero", f.rate_length(0, 30, 24), 0)
check("no fps at all does not divide by zero", f.rate_length(300, 0, 24), 0)

print("select_every_nth DROPS frames, so the count rounds UP")
# every 2nd of 5 keeps frames 0, 2 and 4 -- three of them, not two.
check("every 2nd of 5 keeps 3", f.output_length(5, 24, 0, 2), 3)
check("every 2nd of 240 keeps 120", f.output_length(240, 24, 0, 2), 120)
check("every 3rd of 240 keeps 80", f.output_length(240, 24, 0, 3), 80)
check("step 0 means every frame", f.output_length(240, 24, 0, 0), 240)
check("step 1 means every frame", f.output_length(240, 24, 0, 1), 240)

print("and the two compose in that order: rate first, then selection")
check("300@30 -> 24fps -> every 2nd = 120",
      f.output_length(300, 30, 24, 2), 120)
check("300@30 -> 24fps alone = 240", f.output_length(300, 30, 24, 1), 240)

print("dropping frames makes the clip play FASTER unless the rate drops too")
# the surviving frames are further apart in time; reporting rate/step is what
# keeps a downstream `frames / fps` duration honest
close("24 fps every 2nd reports 12", f.output_fps(24, 0, 2), 12.0)
close("24 fps every frame reports 24", f.output_fps(24, 0, 1), 24.0)
close("force_rate wins over the source", f.output_fps(30, 24, 1), 24.0)
close("and both apply together", f.output_fps(30, 24, 2), 12.0)

print("an OUTPUT frame maps back to a SOURCE time — the only conversion that "
      "touches the file")
close("output 0 is time 0", f.source_time(0, 30, 24, 1), 0.0)
close("output 24 at 24fps is 1.0s", f.source_time(24, 30, 24, 1), 1.0)
# with every_nth, output frame 24 is rate frame 48, which is 2s in
close("every 2nd doubles the stride", f.source_time(24, 30, 24, 2), 2.0)
close("no force_rate uses the source rate", f.source_time(30, 30, 0, 1), 1.0)

print("a span clamps INTO the clip, and is never silently shifted")
check("a span inside is untouched", f.clamp_span(10, 50, 240), (10, 50))
# CUT, NEVER SHIFTED. Returning (190, 50) for a 50-frame ask at 200 would give
# the right LENGTH from the wrong place, which is the worse failure: it renders
# and it is not what was asked for.
check("running past the end is cut short", f.clamp_span(200, 50, 240), (200, 40))
check("count 0 means to the end", f.clamp_span(100, 0, 240), (100, 140))
check("a start past the end pins to the last frame",
      f.clamp_span(999, 10, 240), (239, 1))
check("negatives clamp to zero", f.clamp_span(-5, 10, 240), (0, 10))
check("an empty clip yields an empty span", f.clamp_span(0, 10, 0), (0, 0))

print("core's frame count is cross-checked, because it returns 1 for some files")
# THE BUG BEHIND A TRIM THAT COLLAPSED TO ONE FRAME (CJ, 2026-09-13, two
# different 3-second 30fps clips). ComfyUI's get_frame_count() falls through to
# a decode-and-count loop whose end timestamp is 0, so it breaks on the first
# frame and returns 1 -- for any untrimmed file whose stream carries no frame
# count and no stream duration. Every span then clamps to 1 and cannot be typed
# out of, because 1 is genuinely the whole clip as far as the node knows.
check("a 3s 30fps clip reported as 1 frame is corrected",
      f.sane_frame_count(1, 3.0, 30.0), 90)
check("and a 9.42s 24fps one", f.sane_frame_count(1, 9.42, 24.0), 226)
# ONLY 1 IS OVERRIDDEN: that is the exact signature of the broken branch, and
# second-guessing a good count would be its own bug
check("a plausible count is trusted", f.sane_frame_count(226, 9.42, 24.0), 226)
check("even a small one", f.sane_frame_count(5, 0.2, 24.0), 5)
# a genuinely one-frame file has a duration that AGREES, and is left alone
check("a real one-frame clip survives", f.sane_frame_count(1, 0.033, 30.0), 1)
check("no duration means nothing to check against",
      f.sane_frame_count(1, 0, 30.0), 1)
check("nor does no fps", f.sane_frame_count(1, 3.0, 0), 1)
check("zero stays zero", f.sane_frame_count(0, 0, 0), 0)

print("a span whose CLIP changed keeps its LENGTH and slides")
# THE STUCK-AT-1 BUG, 2026-09-13. Stepping through a folder with a 120-frame
# span, clamp_span hit a clip shorter than the span's START, pinned the start to
# the last frame and cut the count to 1 -- and that got WRITTEN BACK, so every
# later clip inherited a 1-frame span that looked unchangeable.
check("clamp_span, the one-edit rule, is what did it",
      f.clamp_span(100, 120, 60), (59, 1))
check("refit keeps as much length as the clip allows",
      f.refit_span(100, 120, 60), (0, 60))
check("and does not poison the next, longer clip",
      f.refit_span(*f.refit_span(100, 120, 60), 300), (0, 60))
check("a span that merely overruns keeps its start",
      f.refit_span(10, 20, 60), (10, 20))
check("one near the end slides just enough to fit",
      f.refit_span(50, 20, 60), (40, 20))
check("count 0 still means the whole clip", f.refit_span(0, 0, 60), (0, 60))
check("an unknown clip changes nothing at all",
      f.refit_span(100, 120, 0), (100, 120))

print("spans are lines, not json — a person edits them before the timeline "
      "exists to draw them")
s = f.parse_spans("0 192 opening\n240 192 the turn\n", 600)
check("two spans", len(s), 2)
check("the first", (s[0]["start"], s[0]["count"], s[0]["label"]),
      (0, 192, "opening"))
check("the second", (s[1]["start"], s[1]["count"], s[1]["label"]),
      (240, 192, "the turn"))
check("a label is optional",
      f.parse_spans("5 10", 600)[0]["label"], "")
check("comments and blank lines are ignored",
      len(f.parse_spans("# a note\n\n0 10\n", 600)), 1)
check("a malformed line is skipped rather than fatal",
      len(f.parse_spans("not a span\n0 10\n", 600)), 1)
# a freshly dropped video has no spans yet and must still be usable
check("no spans at all yields one covering everything",
      (lambda x: (x[0]["start"], x[0]["count"]))(f.parse_spans("", 600)),
      (0, 600))
check("and spans are clamped on the way in",
      f.parse_spans("500 999", 600)[0]["count"], 100)

print("and they round-trip, because the timeline writes them back")
text = "0 192 opening\n240 192 the turn"
check("format is the inverse of parse",
      f.format_spans(f.parse_spans(text, 600)), text)
check("a span with no label loses no information",
      f.format_spans(f.parse_spans("5 10", 600)), "5 10")

print("frame selection IS rate conversion — one index, not two passes")
# output i -> rate frame i*step -> source frame round(i*step * src/rate)
check("same rate, every frame, is the identity",
      f.select_indices(240, 24, 24, 1, 5), [0, 1, 2, 3, 4])
# 30 -> 24 drops every fifth: source 0,1,2,3,5,6,...  (ratio 1.25)
check("30 -> 24 fps drops every fifth frame",
      f.select_indices(300, 30, 24, 1, 6), [0, 1, 3, 4, 5, 6])
# CONVERTING UP REPEATS, and the repeat is kept: it is what the model sees,
# and de-duplicating would quietly shorten the clip
check("24 -> 30 fps repeats a frame rather than inventing one",
      f.select_indices(240, 24, 30, 1, 6), [0, 1, 2, 2, 3, 4])
check("every 2nd doubles the stride",
      f.select_indices(240, 24, 0, 2, 4), [0, 2, 4, 6])
check("and both compose",
      f.select_indices(300, 30, 24, 2, 4), [0, 3, 5, 8])
check("it never indexes past the window",
      f.select_indices(3, 24, 24, 1, 10), [0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
check("an empty window gives nothing", f.select_indices(0, 24, 24, 1, 5), [])
check("wanting nothing gives nothing", f.select_indices(100, 24, 24, 1, 0), [])
check("as many indices as frames asked for",
      len(f.select_indices(300, 30, 24, 1, 192)), 192)

print("the description reports both clocks")
d = f.describe(300, 30, 24, 1, {"start": 0, "count": 192})
ok("it names the source", "source 300 frames @ 30 fps" in d)
ok("it names the converted output", "output 240 frames @ 24 fps" in d)
ok("it names the span in frames and seconds", "192 frames (8.00s)" in d)
ok("and it stays quiet about conversion when there is none",
   "output" not in f.describe(300, 24, 0, 1, {"start": 0, "count": 100}))

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    sys.exit(1)
print("frames: three domains, one conversion, and the timeline is in output")
