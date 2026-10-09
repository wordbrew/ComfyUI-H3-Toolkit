// Every chunk-editor field must WRITE BACK, not just display.
//
// CJ: "I typed changes to 'What happens in this chunk' and it isn't saving
// it..." Every i_* control was read INTO by drawInspector and read FROM by
// nothing — the handlers existed when the editor was a side panel and were
// lost moving it into a modal. The textarea changed; the take did not.
//
// This is the cheapest possible check for a whole class of bug: a control
// that exists in the markup and in no writer. It reads the page, so it cannot
// pass against a stale copy.
const fs = require("fs");
const html = fs.readFileSync(__dirname + "/ui/index.html", "utf8");
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];

let bad = 0;
const ck = (label, cond, detail = "") => {
  console.log(`  ${cond ? "ok  " : "FAIL"} ${label}${detail ? ": " + detail : ""}`);
  if (!cond) bad++;
};

// every i_* control the markup declares, minus the containers that hold them
const declared = [...new Set(
  [...html.matchAll(/<(?:input|select|textarea)[^>]*\bid="(i_[a-z_]+)"/g)]
    .map(m => m[1]))].sort();
console.log(`the chunk editor declares: ${declared.join(", ")}`);

const table = src.slice(src.indexOf("const INSPECTOR = ["));
const body = table.slice(0, table.indexOf("\n  ];"));

for (const id of declared) {
  // a field is either in the writer table, or wired by its own handler
  const inTable = body.includes(`"${id}"`);
  const own = new RegExp(`\\$\\("${id}"\\)\\.(onclick|onchange|oninput|addEventListener)`)
    .test(src);
  ck(`${id} writes back`, inTable || own,
     inTable ? "in the writer table" : own ? "has its own handler" : "NOTHING READS IT");
}

console.log("\nand the writers actually assign to the chunk");
for (const field of ["beat", "frames", "continuity", "seed_offset"]) {
  ck(`sc.${field} is assigned`, new RegExp(`sc\\.${field}\\s*=`).test(body));
}
ck("each writer then saves", (body.match(/\n/g) || []).length > 0
   && /sceneChanged\(\)/.test(src.slice(src.indexOf("const INSPECTOR = ["),
                                        src.indexOf("const INSPECTOR = [") + 1600)));

console.log("\nand handle completion is attached, not merely defined");
ck("attachHandles is called", /for \(const id of \[[^\]]*\]\) \{\s*attachHandles/.test(src));
ck("on the take's head and tail", /attachHandles[\s\S]{0,200}/.test(src)
   && /\["head", "tail", "i_beat"/.test(src));

console.log();
if (bad) { console.log(`FAIL — ${bad} check(s)`); process.exit(1); }
console.log("ui inspector: every field that is shown is also read back");
