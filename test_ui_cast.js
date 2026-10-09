// The UI's cast ordering must match takeapi's, EXACTLY.
//
// Subject numbering happens on the server (takeapi.cast_order -> h3script) and
// PICTURE numbering happens in this page (deriveRefs -> ref_images.ref_image_N).
// Two passes over one list. If they sort differently, <Subject 1> is Lily while
// <Picture 1> is the lamp, and the model is told a face belongs to a lamp —
// silently, because both halves are internally consistent.
//
// So the orderings are generated here from the same cases and compared.
const fs = require("fs");
const { execFileSync } = require("child_process");

const src = fs.readFileSync(__dirname + "/ui/index.html", "utf8")
  .match(/<script>([\s\S]*?)<\/script>/)[1];
const pick = (n) => {
  const m = src.match(new RegExp(`(?:const|function) ${n}[\\s\\S]*?\\n\\}`));
  if (!m) { console.error(`FAIL cannot find ${n} in ui/index.html`); process.exit(1); }
  return m[0];
};
const { castOrder, castKind } = new Function(
  `${src.match(/const CAST_KIND_ORDER = \{[^}]*\};/)[0]}
   ${pick("castKind")}\n${pick("castOrder")}
   return { castOrder, castKind };`)();

const CASES = [
  [{ key: "lamp", subkind: "thing" }, { key: "room", setting: "a room" },
   { key: "lily", describe: "a woman" }],
  [{ key: "a", describe: "one" }, { key: "b", describe: "two" }],
  [{ key: "room", setting: "x" }],
  [],
  [{ key: "p1", subkind: "thing" }, { key: "p2", subkind: "thing" },
   { key: "c", describe: "c" }, { key: "s", kind: "setting", setting: "s" }],
  // a library row carries neither subkind nor setting: it is a person
  [{ key: "fromlib", from_library: "jen", kind: "library" },
   { key: "t", subkind: "thing" }],
];

const mine = CASES.map(c => castOrder(c).map(r => r.key));
const theirs = JSON.parse(execFileSync("python3", ["-c", `
import importlib.util, json, sys, types
sys.modules.setdefault("torch", types.ModuleType("torch"))
pkg = types.ModuleType("h3t"); pkg.__path__ = ["."]; sys.modules["h3t"] = pkg
def load(n):
    sp = importlib.util.spec_from_file_location("h3t." + n, n + ".py")
    m = importlib.util.module_from_spec(sp); sys.modules["h3t." + n] = m
    sp.loader.exec_module(m); return m
for m in ("timing","chunkplan","prompt_lint","character","h3script","chunklora","chunkref"):
    load(m)
ta = load("takeapi")
cases = json.loads(sys.argv[1])
print(json.dumps([[r["key"] for r in ta.cast_order(c)] for c in cases]))
`, JSON.stringify(CASES)], { cwd: __dirname, encoding: "utf8" }).trim());

let bad = 0;
const ck = (label, got, want) => {
  const o = JSON.stringify(got) === JSON.stringify(want);
  console.log(`  ${o ? "ok  " : "FAIL"} ${label}: ${JSON.stringify(got)}`);
  if (!o) { bad++; console.log(`       wanted ${JSON.stringify(want)}`); }
};
CASES.forEach((c, i) => {
  const ok = JSON.stringify(mine[i]) === JSON.stringify(theirs[i]);
  console.log(`  ${ok ? "ok  " : "FAIL"} case ${i + 1}: ${JSON.stringify(mine[i])}`);
  if (!ok) { bad++; console.log(`       python said ${JSON.stringify(theirs[i])}`); }
});

console.log("\nand the invariant the whole thing exists for");
const o = castOrder(CASES[0]);
const first = castKind(o[0]);
console.log(`  ${first === "person" ? "ok  " : "FAIL"} subject 1 is a person: ${first}`);
if (first !== "person") bad++;
const kinds = castOrder(CASES[4]).map(castKind);
const sorted = [...kinds].sort((a, b) =>
  ({ person: 0, place: 1, thing: 2 })[a] - ({ person: 0, place: 1, thing: 2 })[b]);
const grouped = JSON.stringify(kinds) === JSON.stringify(sorted);
console.log(`  ${grouped ? "ok  " : "FAIL"} people, then place, then things: ${kinds.join(" ")}`);
if (!grouped) bad++;
const two = castOrder([{ key: "a", describe: "1" }, { key: "b", describe: "2" },
                       { key: "t", subkind: "thing" }]).map(r => r.key);
const stable = two[0] === "a" && two[1] === "b";
console.log(`  ${stable ? "ok  " : "FAIL"} adding a prop does not renumber the people: ${two.join(" ")}`);
if (!stable) bad++;

console.log("\nA LIBRARY SUBJECT'S HANDLE COMES FROM ITS NAME");
// CJ: "shouldn't pre-saved subjects have pre-saved @handles? only newly
// created/hand-written subjects should have those added?" Asking someone to
// invent a handle for a thing they already named gives one subject two names,
// and two names drift.
let CASTREF = [];
const H = new Function("getCAST", `
  ${src.match(/function slugHandle[\s\S]*?\n\}/)[0]}
  ${src.match(/function uniqueHandle[\s\S]*?\n\}/)[0].replace("CAST.filter", "getCAST().filter")}
  return { slugHandle, uniqueHandle };`)(() => CASTREF);

ck("a name with a space", H.slugHandle("living room"), "livingroom");
ck("case is dropped", H.slugHandle("Lily"), "lily");
ck("underscores survive, because a filename may carry one",
   H.slugHandle("jen_s"), "jen_s");
ck("punctuation does not", H.slugHandle("Jen's room!"), "jensroom");
ck("an empty name still yields something usable",
   H.uniqueHandle("", "x"), "subject");

console.log("\nand two subjects never share one handle");
CASTREF = [{ id: "1", key: "lily" }, { id: "2", key: "lily2" }];
ck("a clash gets the next free suffix", H.uniqueHandle("Lily", "3"), "lily3");
ck("but editing the row that OWNS the handle keeps it",
   H.uniqueHandle("Lily", "1"), "lily");
CASTREF = [];
ck("no clash, no suffix", H.uniqueHandle("Lily", "9"), "lily");

console.log();
if (bad) { console.log(`FAIL — ${bad} check(s)`); process.exit(1); }
console.log("ui cast: the page and takeapi order the cast identically");
