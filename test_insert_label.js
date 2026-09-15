// The blend ladder, computed twice — once in Python for the render and once in
// JS so the widget's arrows only ever land on values that exist.
//
// This pack's recurring failure is two implementations of one rule. If the
// arrows walk 0, 4, 8 while the node rounds to 0, 1, 5, the number on screen is
// not the number rendered and nothing errors. insert_ladder.json is generated
// FROM THE NODE, so the JS cannot drift without this failing.
const fs = require("fs");
const GROUP = 17, STARTS = [0, 1, 5, 9, 13];
const framesAtStep = (s) => {
  const q = Math.floor(Math.max(0, s) / 5);
  return q * GROUP + STARTS[Math.max(0, s) % 5];
};
const stepAtFrame = (f) => {
  f = Math.max(0, Math.floor(f));
  const g = Math.floor(f / GROUP), r = f % GROUP;
  let k = 0;
  for (let i = 0; i < STARTS.length; i++) if (STARTS[i] <= r) k = i;
  return g * 5 + k;
};
function blendLadder(cutStep, dir, span) {
  const here = framesAtStep(cutStep);
  const out = [];
  for (let n = 0; n <= span; n++) {
    const other = cutStep + dir * n;
    if (other < 0) break;
    out.push(Math.abs(framesAtStep(other) - here));
  }
  return out;
}

const want = JSON.parse(fs.readFileSync(__dirname + "/insert_ladder.json", "utf8"));
let bad = 0;
for (const [split, rungs] of Object.entries(want)) {
  const got = blendLadder(stepAtFrame(Number(split)), -1, 400).slice(0, rungs.length);
  if (JSON.stringify(got) !== JSON.stringify(rungs)) {
    console.log(`  FAIL cut ${split}: arrows walk ${got} but the node renders ${rungs}`);
    bad++;
  }
}
// and the arrows themselves: pressing up from 0 must climb the ladder, never
// sit still and never skip a rung
const L = blendLadder(stepAtFrame(102), -1, 400);
let v = 0, last = 0;
const walked = [0];
for (let i = 0; i < 6; i++) {
  const asked = v + 1;
  v = asked > last ? (L.find((x) => x > last) ?? last) : v;
  last = v;
  walked.push(v);
}
if (JSON.stringify(walked) !== JSON.stringify([0, 4, 8, 12, 16, 17, 21])) {
  console.log("  FAIL six presses of the up arrow gave", walked);
  bad++;
}
console.log();
if (bad) { console.log(`${bad} failure(s)`); process.exit(1); }
console.log(`insert ladder: arrows and node agree at ${Object.keys(want).length} cuts,`
            + ` and six presses give ${walked.join(", ")}`);
