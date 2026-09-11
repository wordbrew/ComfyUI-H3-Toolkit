"""serialize() — the document back to text, which is the half an editor writes.

WHAT ROUND-TRIPPING MEANS
  `parse(serialize(doc)) == doc`, NOT `serialize(parse(text)) == text`. Comments,
  blank lines and the author's spacing are not in the document, so they cannot
  come back. The DOCUMENT is what has to survive, because that is what a panel
  edits and hands back.

WHY THE OVERRIDE RULE IS TESTED HARDEST
  A character's pictures, audio, description and retention come from the store.
  Writing them back into the script would freeze today's values, and re-saving
  the character would silently stop taking effect. serialize() re-parses each
  head line on its own and emits only what differs -- so what is NOT written is
  as much the contract as what is.

    python3 test_script_roundtrip.py
"""
import importlib.util
import pathlib
import sys

_root = pathlib.Path(__file__).parent.resolve()
_spec = importlib.util.spec_from_file_location("h3s", _root / "h3script.py")
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)

fails = []


def check(label, got, want):
    if got != want:
        fails.append(label)
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def roundtrips(label, text):
    doc = hs.parse(text)
    again = hs.parse(hs.serialize(doc))
    check(label, again, doc)
    return doc


print("the document survives a trip through text")
roundtrips("a minimal take", """
@ada = a woman with red hair
shot | she stands at the window
""")

roundtrips("notes, actions and dialogue", """
@ada = a woman with red hair
@room = setting. a bedroom in warm low light
shot | a two-shot on the bed
  note @ada is nearer camera
  do   she turns her head
  say  @ada moans Oh god, yes.
""")

roundtrips("several shots and a chunk split", """
@ada = a woman with red hair
@man = a man in his thirties
shot | the first shot
  say  @ada Hello.
  ---
  say  @man Goodbye.
shot | the second shot
  do   she crosses the room
""")

roundtrips("settings that are not the defaults", """
@ada = a woman with red hair
task = video editing
soundscape = rain on glass
music = a slow piano
shot | she stands at the window
""")

roundtrips("cast attribute overrides", """
@ada = a woman with red hair
@ada.retention = free
@ada.preserve = build and hair
@ada.allow = expression and posture
shot | she stands at the window
""")

roundtrips("a lora with a two-number strength", """
@ada = a woman with red hair
shot | she stands at the window
  lora h3/Thing.safetensors 0.8 1.0
""")

print("shot lengths and per-shot loras survive the round trip")
roundtrips("a shot with a length and a lora", """
@ada = a woman
shot 141 | she stands at the window
  lora h3/Thing.safetensors 0.8
  say @ada hi
""")
roundtrips("a placed line keeps its time", """
@ada = a woman
shot 243 | she crosses the room
  say @ada @2.5 asks Do you want to play?
""")
doc = hs.parse("@ada = a woman\nshot 141 | s\n  say @ada @2.5 hi\n")
check("the time is on the line, in seconds",
      doc["shots"][0]["chunks"][0]["lines"][0]["at"], 2.5)
check("and it is written back as @2.5", "@ada @2.5" in hs.serialize(doc), True)
check("the shot's length is written back", "shot 141 |" in hs.serialize(doc), True)

print("defaults are left out rather than written back")
doc = hs.parse("@ada = a woman with red hair\nshot | a shot\n")
text = hs.serialize(doc)
check("no task line when it is the default", "task =" in text, False)
check("no music line when it is the default", "music =" in text, False)
check("no empty soundscape line", "soundscape =" in text, False)
check("no retention line when it was never overridden",
      "retention =" in text, False)

print("a speech verb survives, and a missing one becomes 'says'")
doc = hs.parse("@ada = a woman\nshot | s\n  say @ada whispers come here\n")
check("whispers is kept", "whispers come here" in hs.serialize(doc), True)
doc = hs.parse("@ada = a woman\nshot | s\n  say @ada come here\n")
check("a bare line is written as says", "says come here" in hs.serialize(doc), True)
check("and it still round-trips", hs.parse(hs.serialize(doc)), doc)

print("how a line is said is a PHRASE, not one of six words")
roundtrips("a piped verb phrase", """
@ada = a woman with red hair
shot | she stands at the window
  say @ada | says quietly, half-turning away | I told you I would come back.
""")
doc = hs.parse("@ada = a woman\nshot | s\n"
               "  say @ada | almost laughing | Do you want to play?\n")
line = doc["shots"][0]["chunks"][0]["lines"][0]
check("the whole phrase is the verb", line["verb"], "almost laughing")
check("and none of it leaks into the spoken line",
      line["line"], "Do you want to play?")
check("a phrase is written back piped", "| almost laughing |" in hs.serialize(doc), True)

doc = hs.parse("@ada = a woman\nshot | s\n  say @ada whispers come here\n")
check("a bare verb stays bare", "@ada whispers come here" in hs.serialize(doc), True)

print("a pipe inside the spoken line does not break the round trip")
roundtrips("a line containing a pipe", """
@ada = a woman
shot | s
  say @ada | says | she typed a || b and laughed
""")

print("serialize does not need a store, a graph or torch")
check("output is text", isinstance(hs.serialize(hs.parse(
    "@ada = a woman\nshot | s\n")), str), True)
check("an empty document is text, not a crash", hs.serialize({}), "\n")

print()
print("NOT COVERED: the `character <Name>` head line, because it reads the")
print("character store and there is none in WSL. The override rule is exercised")
print("through plain subjects, which take the same code path.")
print()
print(f"{len(fails)} failure(s)" if fails else "script round trip: all checks pass")
sys.exit(1 if fails else 0)
