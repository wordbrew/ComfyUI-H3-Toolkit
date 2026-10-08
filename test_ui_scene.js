// Duplicating a scene must not alias the original.
//
// CJ: "when we choose to duplicate a scene in the ui, it permanently ties the
// duplicated scene details to the original. As in, if I click duplicate off of
// Scene 2, into scene 3. Then I pick a new lora in scene 3 for use, it will
// also change the lora for scene 2."
//
// `{ ...src }` copies one level: `frames` and `beat` by value, and the SAME
// `loras` array reference into both scenes. The function is read out of
// ui/index.html so this cannot pass against a stale copy.
const fs = require("fs");
const src = fs.readFileSync(__dirname + "/ui/index.html", "utf8")
  .match(/<script>([\s\S]*?)<\/script>/)[1];
const body = src.match(/function cloneScene\(src\) \{[\s\S]*?\n\}/)[0];
const cloneScene = new Function(`${body}; return cloneScene;`)();
const a = { frames: 192, beat: "x", loras: [{ name: "n.safetensors", strength: 0.8 }] };
const b = cloneScene(a);
b.loras[0].name = "other.safetensors";
b.loras.push({ name: "second", strength: 1 });
b.frames = 141;
let bad = 0;
const ck = (l, g, w) => { const o = JSON.stringify(g) === JSON.stringify(w);
  console.log(`  ${o ? "ok  " : "FAIL"} ${l}: ${JSON.stringify(g)}`); if (!o) bad++; };
ck("the original keeps its LoRA name", a.loras[0].name, "n.safetensors");
ck("and its LoRA count", a.loras.length, 1);
ck("and its length", a.frames, 192);
ck("while the copy has its own", b.loras.length, 2);
process.exit(bad ? 1 : 0);

console.log("ui scene: a duplicate owns its own nested state");
