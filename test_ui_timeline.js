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
  ${grab("covers")}${grab("videoAt")}${grab("audioAt")}${grab("srcTime")}
  ${grab("trackEnd")}
  let SEL2 = new Set();
  let CLIPBOARD = [];
  ${grab("selectedClips")}${grab("trimTo")}${grab("duplicate")}
  ${grab("rippleDelete")}${grab("pasteAt")}${grab("delSelected")}
  const uidq = uid;
  function editChanged() {}
  return { clipLen, clipEnd, projectEnd, snap, blade, pxPerFrame, f2x, x2f,
           covers, videoAt, audioAt, srcTime,
           trimTo, duplicate, rippleDelete, pasteAt, delSelected,
           sel: (ids) => { SEL2 = new Set(ids); },
           board: (cs) => { CLIPBOARD = cs; }, trackEnd };
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

console.log("\nWHAT IS ON SCREEN is a question about the stack");
// Video tracks composite bottom-up and the topmost clip covering the
// playhead wins. Tracks are stored TOP FIRST, so the first match is the
// winner — and that has to be the same order the export composites in, or
// the preview lies about the result.
EDIT = {
  name: "t", playhead: 0, zoom: 0,
  tracks: [
    { id: "v2", kind: "video", name: "V2",
      clips: [clip({ id: "top", start: 48, in: 0, out: 48 })] },
    { id: "v1", kind: "video", name: "V1",
      clips: [clip({ id: "bed", start: 0, in: 0, out: 240 })] },
    { id: "a1", kind: "audio", name: "A1",
      clips: [clip({ id: "mus", start: 0, in: 0, out: 240 })] },
  ],
};
ck("below the overlay, the bed shows", api.videoAt(24).id, "bed");
ck("where they overlap, the TOP track wins", api.videoAt(60).id, "top");
ck("after the overlay, the bed again", api.videoAt(120).id, "bed");
ck("audio plays throughout, regardless",
   api.audioAt(60).map(c => c.id), ["mus"]);
ck("and a gap shows nothing", api.videoAt(900), null);

console.log("\ncovers() is half-open — a clip owns its start, not its end");
// Otherwise two adjacent clips both claim the boundary frame and the winner
// depends on track order rather than on the edit.
ck("owns its first frame", api.covers(clip({ start: 10, out: 10 }), 10), true);
ck("not the frame after its last",
   api.covers(clip({ start: 10, in: 0, out: 10 }), 20), false);
ck("and the one before it is not its", api.covers(clip({ start: 10 }), 9), false);

console.log("\nSOURCE TIME accounts for the trim, not just the position");
// A clip trimmed to start 24 frames in, placed 48 frames along the timeline,
// must show source frame 24 at timeline frame 48 — reading the playhead
// directly would show the wrong part of the file.
const trimmed = clip({ start: 48, in: 24, out: 120 });
ck("its first frame is its IN point", api.srcTime(trimmed, 48), 24 / 24);
ck("and ten frames later is ten frames further in",
   api.srcTime(trimmed, 58), 34 / 24);

console.log("\nRIPPLE DELETE closes the hole — on that track only");
// A ripple across every track would silently re-time footage the editor was
// not looking at. Resolve and Premiere both scope it to the track for that
// reason, and a layered timeline makes the alternative actively dangerous.
EDIT = {
  name: "t", playhead: 0, zoom: 0,
  tracks: [
    { id: "v1", kind: "video", name: "V1", clips: [
      clip({ id: "a", start: 0, in: 0, out: 48 }),
      clip({ id: "b", start: 48, in: 0, out: 48 }),
      clip({ id: "c", start: 96, in: 0, out: 48 }),
    ] },
    { id: "v2", kind: "video", name: "V2", clips: [
      clip({ id: "z", start: 96, in: 0, out: 48 }),
    ] },
  ],
};
api.sel(["b"]);
ck("one clip removed", api.rippleDelete(), 1);
const v1 = EDIT.tracks[0].clips.slice().sort((x, y) => x.start - y.start);
ck("what followed slid back by its length",
   v1.map(c => [c.id, c.start]), [["a", 0], ["c", 48]]);
ck("THE OTHER TRACK DID NOT MOVE", EDIT.tracks[1].clips[0].start, 96);

console.log("\nand a plain delete leaves the hole");
EDIT = {
  name: "t", playhead: 0, zoom: 0,
  tracks: [{ id: "v1", kind: "video", name: "V1", clips: [
    clip({ id: "a", start: 0, in: 0, out: 48 }),
    clip({ id: "b", start: 48, in: 0, out: 48 }),
    clip({ id: "c", start: 96, in: 0, out: 48 }),
  ] }],
};
api.sel(["b"]);
ck("removed", api.delSelected(), 1);
ck("the survivors stay where they were",
   EDIT.tracks[0].clips.map(c => [c.id, c.start]), [["a", 0], ["c", 96]]);

console.log("\nTRIM TO PLAYHEAD moves one edge and keeps the frames aligned");
EDIT = { name: "t", playhead: 24, zoom: 0,
         tracks: [{ id: "v1", kind: "video", name: "V1",
                    clips: [clip({ id: "a", start: 0, in: 10, out: 58 })] }] };
const c0 = EDIT.tracks[0].clips[0];
ck("trimming the start", api.trimTo(c0, "l"), 1);
ck("  the in point advanced by the same amount the start did",
   [c0.in, c0.start], [34, 24]);
ck("  so the frame under the playhead is the one that was there",
   api.srcTime(c0, 24), 34 / 24);

EDIT = { name: "t", playhead: 24, zoom: 0,
         tracks: [{ id: "v1", kind: "video", name: "V1",
                    clips: [clip({ id: "a", start: 0, in: 10, out: 58 })] }] };
const c1 = EDIT.tracks[0].clips[0];
ck("trimming the end", api.trimTo(c1, "r"), 1);
ck("  it now ends at the playhead", api.clipEnd(c1), 24);
ck("  and still starts where it did", [c1.start, c1.in], [0, 10]);

EDIT = { name: "t", playhead: 500, zoom: 0,
         tracks: [{ id: "v1", kind: "video", name: "V1",
                    clips: [clip({ id: "a", start: 0, in: 0, out: 48 })] }] };
ck("a playhead outside the clip trims nothing",
   api.trimTo(EDIT.tracks[0].clips[0], "l"), 0);

console.log("\nDUPLICATE lands after the original, not on top of it");
EDIT = { name: "t", playhead: 0, zoom: 0,
         tracks: [{ id: "v1", kind: "video", name: "V1",
                    clips: [clip({ id: "a", start: 12, in: 0, out: 48 })] }] };
api.sel(["a"]);
ck("one made", api.duplicate(), 1);
const dup = EDIT.tracks[0].clips.find(c => c.id !== "a");
ck("it starts where the original ends", dup.start, 60);
ck("and has its own identity", dup.id !== "a", true);

console.log("\nPASTE keeps the relative spacing of what was copied");
EDIT = { name: "t", playhead: 0, zoom: 0,
         tracks: [{ id: "v1", kind: "video", name: "V1", clips: [] }] };
const board = [clip({ id: "p", start: 100, in: 0, out: 24 }),
               clip({ id: "q", start: 160, in: 0, out: 24 })];
api.board(board);
ck("two pasted", api.pasteAt(EDIT.tracks[0], 10), 2);
const pasted = EDIT.tracks[0].clips.slice().sort((a, b) => a.start - b.start);
ck("the first lands at the playhead", pasted[0].start, 10);
ck("and the gap between them survives",
   pasted[1].start - pasted[0].start, 60);

console.log();
if (bad) { console.log(`FAIL — ${bad} check(s)`); process.exit(1); }
console.log("ui timeline: frames, gaps, splits and snapping all hold");

// ---------------------------------------------------------------------------
console.log("\nA DROPPED CLIP APPENDS TO ITS OWN TRACK");
// CJ: "if no video is there, then it drops at the start for that layer. if
// video is there, then it snaps and drops to the end." PER TRACK — each
// layer fills from its own start, so dropping a run onto V1 builds a
// sequence instead of a pile wherever the cursor happened to be.
EDIT = { name: "t", playhead: 0, zoom: 0, tracks: [
  { id: "v1", kind: "video", name: "V1", clips: [] },
  { id: "v2", kind: "video", name: "V2", clips: [
      clip({ id: "x", start: 0, in: 0, out: 48 }),
      clip({ id: "y", start: 48, in: 0, out: 72 }) ] },
] };
ck("an empty track starts at zero", api.trackEnd(EDIT.tracks[0]), 0);
ck("a filled one appends after its last clip",
   api.trackEnd(EDIT.tracks[1]), 120);
ck("and it is the LAST end, not the clip count",
   api.trackEnd({ clips: [clip({ start: 500, in: 0, out: 24 }),
                          clip({ start: 0, in: 0, out: 24 })] }), 524);
ck("a gap before the last clip does not shrink it",
   api.trackEnd({ clips: [clip({ start: 240, in: 0, out: 48 })] }), 288);

console.log("\nand each layer is independent — one track filling does not " +
            "move another");
ck("V1 still starts at zero while V2 is full",
   [api.trackEnd(EDIT.tracks[0]), api.trackEnd(EDIT.tracks[1])], [0, 120]);

console.log("\nA CONTEXT MENU ITEM MUST SURVIVE BEING PRESSED");
// The first version dismissed on ANY mousedown. mousedown fires before click,
// so pressing an item removed the element out from under the click that was
// about to run it — every item in the menu was inert, and the two CJ happened
// to try were the two he reported. A menu that closes before its own button
// fires is indistinguishable from a menu whose handlers are wrong.
{
  let bad2 = 0;
  const ck2 = (l, c) => {
    console.log(`  ${c ? "ok  " : "FAIL"} ${l}`);
    if (!c) bad2++;
  };
  const open = src.slice(src.indexOf("function openMenu"));
  const body = open.slice(0, open.indexOf("\nfunction "));
  ck2("dismissal ignores clicks inside the menu",
      /m\.contains\(ev\.target\)/.test(body));
  ck2("and does not blanket-close on mousedown",
      !/addEventListener\("mousedown",\s*closeMenu/.test(body));
  ck2("closeMenu removes its own listener, so menus do not leak one each",
      /removeEventListener\("mousedown"/.test(
        src.slice(src.indexOf("function closeMenu"),
                  src.indexOf("function closeMenu") + 400)));
  if (bad2) { console.log(`FAIL — ${bad2} check(s)`); process.exit(1); }
}
console.log("ui timeline: and the menu can actually be used");
