// The cutting room's arithmetic, read out of the page.
//
// CJ: "I want this like DaVinci or premiere. timeline editing." A timeline is
// POSITIONAL: a clip knows where it starts, so gaps exist and two clips on
// different tracks can overlap. That is a different model from an ordered
// list, and the places it goes wrong are all arithmetic — a trim that moves
// the wrong edge, a split that loses or duplicates a frame, a snap that fires
// at the wrong zoom.
//
// Everything is integer FRAMES. Seconds would put a rounding step between the
// edit and the render, and `spans` — which VideoTrimLoad already takes — is
// frames.
const fs = require("fs");
const src = fs.readFileSync(__dirname + "/ui/index.html", "utf8")
  .match(/<script>([\s\S]*?)<\/script>/)[1];
const grab = (n) => {
  const m = src.match(new RegExp(`function ${n}\\([\\s\\S]*?\\n\\}`));
  if (!m) { console.error(`FAIL cannot find ${n}()`); process.exit(1); }
  return m[0];
};

let EDIT = null;
const SNAP = { checked: true };
const api = new Function("getEDIT", "setEDIT", "snapOn", `
  const FPS2 = 24;
  const uid = () => Math.random().toString(36).slice(2, 9);
  const $ = (id) => (id === "ed_snap" ? snapOn : { value: "", textContent: "" });
  Object.defineProperty(globalThis, "EDIT",
    { get: getEDIT, set: setEDIT, configurable: true });
  ${grab("pxPerFrame")}
  const f2x = (f) => f * pxPerFrame();
  const x2f = (x) => Math.max(0, Math.round(x / pxPerFrame()));
  ${grab("clipLen")}${grab("clipEnd")}${grab("projectEnd")}
  ${grab("allClips")}${grab("snap")}${grab("blade")}
  function editChanged() {}
  return { clipLen, clipEnd, projectEnd, snap, blade, pxPerFrame, f2x, x2f };
`)(() => EDIT, (v) => { EDIT = v; }, SNAP);

let bad = 0;
const ck = (l, g, w) => {
  const o = JSON.stringify(g) === JSON.stringify(w);
  console.log(`  ${o ? "ok  " : "FAIL"} ${l}: ${JSON.stringify(g)}`);
  if (!o) { bad++; console.log(`       wanted ${JSON.stringify(w)}`); }
};

const clip = (o) => Object.assign(
  { id: o.id || "c", source: "a.mp4", in: 0, out: 96, start: 0 }, o);
const proj = (clips, zoom = 0) => ({
  name: "t", playhead: 0, zoom,
  tracks: [{ id: "v1", kind: "video", name: "V1", clips }],
});

console.log("a clip's length is its TRIM, not its source length");
EDIT = proj([clip({ in: 24, out: 120 })]);
ck("96 frames of a longer file", api.clipLen(EDIT.tracks[0].clips[0]), 96);
ck("and it ends where it starts plus that",
   api.clipEnd(clip({ start: 240, in: 24, out: 120 })), 336);

console.log("\ngaps are real — the project ends at the LAST clip's end");
EDIT = proj([clip({ id: "a", start: 0, out: 48 }),
             clip({ id: "b", start: 480, out: 48 })]);
ck("not the sum of the lengths", api.projectEnd(), 528);

console.log("\nTHE BLADE splits without copying media");
EDIT = proj([clip({ id: "a", in: 0, out: 96, start: 0 })]);
EDIT.playhead = 36;
ck("one cut makes two clips", api.blade(), 1);
const cs = EDIT.tracks[0].clips.slice().sort((x, y) => x.start - y.start);
ck("the halves are adjacent on the timeline",
   [cs[0].start, api.clipEnd(cs[0]), cs[1].start], [0, 36, 36]);
ck("and adjacent in the SOURCE — no frame lost or repeated",
   [cs[0].in, cs[0].out, cs[1].in, cs[1].out], [0, 36, 36, 96]);
ck("total length is unchanged",
   api.clipLen(cs[0]) + api.clipLen(cs[1]), 96);
ck("both still point at one file",
   [cs[0].source, cs[1].source], ["a.mp4", "a.mp4"]);

console.log("\nthe blade ignores what it does not cross");
EDIT = proj([clip({ id: "a", start: 0, out: 48 })]);
EDIT.playhead = 48;                       // exactly the end — not inside
ck("a playhead on the boundary splits nothing", api.blade(), 0);
EDIT.playhead = 500;
ck("and neither does one past everything", api.blade(), 0);

console.log("\nSNAPPING is to edges and the playhead, in pixels not frames");
EDIT = proj([clip({ id: "a", start: 0, out: 96 })]);
EDIT.playhead = 300;
ck("a near miss lands on the clip's end", api.snap(94, "b"), 96);
ck("and on the playhead", api.snap(302, "b"), 300);
ck("a clip does not snap to itself", api.snap(94, "a"), 94);
ck("far away is left alone", api.snap(400, "b"), 400);
SNAP.checked = false;
ck("and snapping off means off", api.snap(94, "b"), 94);
SNAP.checked = true;

console.log("\nthe snap threshold holds at every zoom, because it is pixels");
EDIT = proj([clip({ id: "a", start: 0, out: 96 })], 0);
const near0 = api.snap(96 - Math.round(4 / api.pxPerFrame()), "b");
EDIT = proj([clip({ id: "a", start: 0, out: 96 })], 30);
const near30 = api.snap(96 - Math.round(4 / api.pxPerFrame()), "b");
ck("snaps when 4px away, zoomed out", near0, 96);
ck("snaps when 4px away, zoomed in", near30, 96);

console.log("\nzoom is one multiplier and always positive");
EDIT = proj([], -40);
const lo = api.pxPerFrame();
EDIT = proj([], 40);
const hi = api.pxPerFrame();
ck("zooming in magnifies", hi > lo * 50, true);
ck("and never reaches zero", lo > 0, true);

console.log();
if (bad) { console.log(`FAIL — ${bad} check(s)`); process.exit(1); }
console.log("ui timeline: frames, gaps, splits and snapping all hold");
