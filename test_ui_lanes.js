// The chunk-span maths behind the reference lanes.
//
// CJ: "default to the whole shot, allow adjustment to specific chunk if you
// want it." So a dropped reference takes the shot it landed in, and a drag
// narrows or widens it — across a cut if that is what you drag.
//
// The three functions are read out of ui/index.html so this cannot pass
// against a stale copy.
const fs = require("fs");
const src = fs.readFileSync(__dirname + "/ui/index.html", "utf8")
  .match(/<script>([\s\S]*?)<\/script>/)[1];
const pick = (name) => {
  const m = src.match(new RegExp(`function ${name}\\([\\s\\S]*?\\n\\}`));
  if (!m) { console.error(`FAIL could not find ${name}() in ui/index.html`); process.exit(1); }
  return m[0];
};
const { specToSet, setToSpec, shotSpan } = new Function(
  `${pick("specToSet")}\n${pick("setToSpec")}\n${pick("shotSpan")}\n` +
  `return { specToSet, setToSpec, shotSpan };`)();

let bad = 0;
const S = (x) => JSON.stringify([...x].sort((a, b) => a - b));
const ck = (l, g, w) => {
  const o = JSON.stringify(g) === JSON.stringify(w);
  console.log(`  ${o ? "ok  " : "FAIL"} ${l}: ${JSON.stringify(g)}`);
  if (!o) { bad++; console.log(`       wanted ${JSON.stringify(w)}`); }
};

console.log("a spec reads the way the LoRA schedule's does");
ck("all", S(specToSet("all", 4)), S([0, 1, 2, 3]));
ck("empty means all, the old behaviour", S(specToSet("", 4)), S([0, 1, 2, 3]));
ck("a single chunk is 1-based", S(specToSet("2", 4)), S([1]));
ck("a range", S(specToSet("2-4", 4)), S([1, 2, 3]));
ck("a list", S(specToSet("1,3", 4)), S([0, 2]));
ck("open-ended", S(specToSet("3+", 4)), S([2, 3]));
ck("last", S(specToSet("last", 4)), S([3]));
ck("and anything past the end is dropped, not wrapped",
   S(specToSet("9", 4)), S([]));

console.log("\nand writing one back collapses runs");
ck("a run becomes a range", setToSpec(new Set([1, 2, 3]), 5), "2-4");
ck("singles stay single", setToSpec(new Set([0, 2]), 5), "1,3");
ck("two runs", setToSpec(new Set([0, 1, 3, 4]), 5), "1-2,4-5");
ck("everything is `all`", setToSpec(new Set([0, 1, 2]), 3), "all");
ck("nothing is empty", setToSpec(new Set(), 3), "");

console.log("\nround-tripping is stable, or a drag would drift");
for (const spec of ["2-4", "1,3", "all", "last", "3+"]) {
  const once = setToSpec(specToSet(spec, 5), 5);
  ck(`  ${spec}`, setToSpec(specToSet(once, 5), 5), once);
}

console.log("\na dropped reference takes the SHOT it landed in");
//        chunk: 1      2      3(cut)  4      5(cut)
const CH = [{}, {}, { cut: true }, {}, { cut: true }];
ck("dropped on chunk 1 takes chunks 1-2", S(shotSpan(0, CH)), S([0, 1]));
ck("dropped on chunk 2 takes the same shot", S(shotSpan(1, CH)), S([0, 1]));
ck("dropped on the cut starts a new shot", S(shotSpan(2, CH)), S([2, 3]));
ck("dropped on chunk 4 takes that shot", S(shotSpan(3, CH)), S([2, 3]));
ck("a one-chunk shot is one chunk", S(shotSpan(4, CH)), S([4]));
ck("with no cuts at all, the shot is the whole take",
   S(shotSpan(1, [{}, {}, {}])), S([0, 1, 2]));

console.log("\nand what a drop writes is what the node will read");
ck("a shot becomes a spec", setToSpec(shotSpan(2, CH), CH.length), "3-4");

console.log();
if (bad) { console.log(`FAIL — ${bad} check(s)`); process.exit(1); }
console.log("ui lanes: spans survive the round trip, and a drop takes its shot");
