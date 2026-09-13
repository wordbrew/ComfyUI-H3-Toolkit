"""prompt_lint's conflict rules — two sources saying different things.

WHY THESE TWO RULES EXIST

  Several nodes in this pack can produce the same section of a prompt. H3
  Character builds subject_definitions and retention_analysis for ONE saved
  asset; H3 Script builds them for the whole cast; H3 Long-Form Links assembles
  a prompt around a clause. A graph is free to wire more than one of them into
  the same prompt, and nothing stops it.

  What comes out is not an error. It is a prompt that defines <Subject 1> twice,
  possibly with different retention markers, and the model picks one. That is
  the exact failure mode this pack was built to end -- five nodes whose fields
  must agree and which nothing checks against each other.

  So the check reads the ASSEMBLED TEXT rather than the graph: it catches the
  conflict whoever produced it, and it needs no node to know about any other.

    python3 test_prompt_lint.py
"""
import importlib.util
import pathlib
import sys

_root = pathlib.Path(__file__).parent.resolve()
_spec = importlib.util.spec_from_file_location("pl", _root / "prompt_lint.py")
pl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pl)

fails = []


def check(label, got, want):
    if got != want:
        fails.append(label)
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


def rules(text, prefix="conflict/"):
    return sorted({r for _, r, _ in pl.lint(text) if r.startswith(prefix)})


CLEAN = """subject_definitions:
<Subject 1> is a woman in her thirties, shown in <Picture 1>.
<Subject 2> is a man with grey hair.

summary:
[reference generation] In a single continuous shot, <Subject 1> speaks 1 line.
The framing holds and the run is unbroken.

retention_analysis:
<Subject 1>: fully_preserved - preserve facial identity as shown in <Picture 1>.
<Subject 2>: partially_preserved - retain the same hair.

detailed_description:
Photorealistic live-action, handheld. [Shot 1] A close-up of <Subject 1>.
<Subject 1> (S1) says: <d>[English] You said tomorrow.</d>

overall_soundscape: close room tone in a hard concrete stairwell.

non_diegetic_music: N/A"""

print("a prompt from one source is clean")
check("no conflict findings", rules(CLEAN), [])

print("the same subject described twice is caught, in either section")
# H3 Character wired into retention_1 and H3 Script into retention_2 is the
# shape this happens in: both name <Subject 1>, with different markers.
two_ret = CLEAN.replace(
    "<Subject 2>: partially_preserved - retain the same hair.",
    "<Subject 2>: partially_preserved - retain the same hair.\n"
    "<Subject 1>: partially_preserved - retain the same build.")
check("two retention lines for one subject",
      rules(two_ret), ["conflict/retention_analysis"])
two_def = CLEAN.replace(
    "<Subject 2> is a man with grey hair.",
    "<Subject 2> is a man with grey hair.\n<Subject 1> is someone else.")
check("two definitions for one subject",
      rules(two_def), ["conflict/subject_definitions"])
ok("the message names the subject and both likely sources",
   all(s in [m for _, r, m in pl.lint(two_ret)
             if r == "conflict/retention_analysis"][0]
       for s in ("<Subject 1>", "H3 Character", "H3 Script")))

print("a prompt assembled around a prompt is caught")
# H3 Script's chunk_prompts ARE complete prompts. Leaving H3 Long-Form Links'
# own head / subject_def_1 / retention_1 filled nests the format inside itself:
# it renders, and it renders wrong.
nested = CLEAN.replace("detailed_description:\nPhotorealistic",
                       "detailed_description:\n" + CLEAN + "\nPhotorealistic")
check("doubled section headings", rules(nested), ["conflict/nested"])
ok("it is an ERROR, not a warning",
   any(s == "ERROR" for s, r, _ in pl.lint(nested) if r == "conflict/nested"))
ok("and it says which heading repeated",
   "subject_definitions" in [m for _, r, m in pl.lint(nested)
                             if r == "conflict/nested"][0])

print("neither rule fires on things that merely LOOK similar")
# a subject mentioned many times in the description is normal prose, not a
# duplicate definition -- the rule reads line STARTS inside the two sections
ok("repeat mentions in the description are fine",
   not rules(CLEAN.replace("<d>[English] You said tomorrow.</d>",
                           "<d>[English] You said tomorrow.</d> "
                           "<Subject 1> (S1) adds: <d>[English] To when?</d>")))
ok("a one-subject prompt is fine",
   not rules("""subject_definitions:
<Subject 1> is a woman.

summary:
[reference generation] One shot.

retention_analysis:
<Subject 1>: fully_preserved - preserve her face.

detailed_description:
[Shot 1] A close-up of <Subject 1>.

overall_soundscape: room tone.

non_diegetic_music: N/A"""))

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    sys.exit(1)
print("prompt lint: two sources cannot disagree without saying so")
