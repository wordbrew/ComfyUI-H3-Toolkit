// THE GALLERIES FROZE THE BROWSER, TWICE, FOR THE SAME REASON.
//
// CJ, first: "the upload/load gallery for content is slow as balls and buggy
// as hell. it will honestly hang up my browser." Then, after a fix that made
// the thumbnails lazy and copied each frame into a <canvas>: "the preview
// tiles for video loading in gallery also doesnt work. slow and wonky and
// causes browser to freeze." And in the same breath: "the video load and trim
// still doesn't work for additional references."
//
// All one bug. His input folder is 371 mp4s and 6.7 GB -- median 11.8 MB, p90
// 44.9 MB. Drawing a tile with a <video preload="metadata"> asks the browser
// to pull metadata from dozens of those at once; it allows about six
// connections, so the rest queue, and an mp4 with its `moov` atom at the end
// means a range request near the tail as well. The page's own requests wait
// behind that. Lazy mounting removed the retained decoders but not the fetch
// storm -- the work is proportional to the bytes.
//
// So the rule this file enforces is absolute and easy to check: A GALLERY
// TILE NEVER CREATES A <video>. The frame comes from /h3_toolkit/thumb, which
// decodes it once with PyAV and caches it (measured on those files: 0.05s and
// ~3.5 KB each, 4ms on a cache hit). A tile is an <img>, same as the image
// tiles beside it.
//
// It also checks the other half of the first fix, which still stands: a
// selection click must not rebuild the gallery.
const fs = require("fs");
const src = fs.readFileSync(__dirname + "/ui/index.html", "utf8")
  .match(/<script>([\s\S]*?)<\/script>/)[1];

function grab(re) {
  const m = src.match(re);
  if (!m) { console.log("FAIL — could not find " + re); process.exit(1); }
  return m[0];
}

let bad = 0;
const ck = (l, got, want) => {
  const o = JSON.stringify(got) === JSON.stringify(want);
  console.log(`  ${o ? "ok  " : "FAIL"} ${l}: ${JSON.stringify(got)}`
    + (o ? "" : ` (want ${JSON.stringify(want)})`));
  if (!o) bad++;
};

/* ---- the DOM, only as much as these functions touch. Every element created
   is recorded by tag, which is what lets the test assert that no <video> was
   ever made rather than trusting a comment that says so. */
const MADE = [];
function el(tag) {
  const listeners = {};
  const classes = new Set();
  const kids = [];
  const e = {
    _tag: tag, tagName: tag.toUpperCase(), _kids: kids,
    style: {}, dataset: {}, textContent: "", alt: "", src: "",
    loading: "", decoding: "", title: "", _html: "",
    get className() { return [...classes].join(" "); },
    set className(v) {
      classes.clear();
      for (const c of String(v).split(/\s+/).filter(Boolean)) classes.add(c);
    },
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      contains: (c) => classes.has(c),
      toggle: (c, on) => (on === undefined
        ? (classes.has(c) ? classes.delete(c) : classes.add(c))
        : (on ? classes.add(c) : classes.delete(c))),
    },
    // the real one REMOVES CHILDREN; a stub that keeps them would let a
    // missing clear pass, which is the shape of the bug being tested
    get innerHTML() { return e._html; },
    set innerHTML(v) { e._html = v; kids.length = 0; },
    addEventListener: (n, fn, opt) => {
      (listeners[n] = listeners[n] || []).push({ fn, once: !!(opt && opt.once) });
    },
    _fire: (n, ev) => {
      const l = listeners[n] || [];
      listeners[n] = l.filter(x => !x.once);
      for (const x of l) x.fn(ev);
    },
    appendChild: (c) => { kids.push(c); return c; },
    append: (...c) => kids.push(...c),
    prepend: (c) => { kids.unshift(c); return c; },
    querySelector: () => null,
    replaceWith: (other) => { e._replacedBy = other; },
  };
  MADE.push(e);
  return e;
}
const document = { createElement: el };

let served = {};
const getJSON = async (u) => served[u];

const env = {
  document, setTimeout, clearTimeout, getJSON,
  API: "/h3_toolkit",
  rawURL: (p) => "/view?" + p,
  escapeHTML: (s) => s,
  isVideoPath: (p) => /\.(mp4|mov|webm)$/i.test(p),
  isAudioPath: (p) => /\.(wav|mp3|flac)$/i.test(p),
  assetURL: (p) => "/view?prev=" + p,
  folderOf: (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : ""),
  FPS2: 24,
};

const body = [
  grab(/function debounce\([\s\S]*?\n\}/),
  grab(/function thumbURL\([\s\S]*?\n\}/),
  grab(/function videoThumb\([\s\S]*?\n\}/),
  grab(/const CLIPLEN = new Map\(\);[\s\S]*?\nasync function clipInfo\([\s\S]*?\n\}/),
  grab(/function drawPicker\(\)[\s\S]*?\nfunction paintPickCount\([\s\S]*?\n\}/),
  grab(/function learnDuration\([\s\S]*?\n\}/),
].join("\n");

const names = Object.keys(env);
const api = new Function(...names, `
  let pickState = null, ASSETS = [], DUR = {};
  let $hooks = {};
  const $ = (id) => ($hooks[id] = $hooks[id] || document.createElement("div"));
  ${body}
  return { videoThumb, drawPicker, learnDuration, debounce, thumbURL, clipInfo,
           $, setPick: (v) => { pickState = v; }, getPick: () => pickState,
           setAssets: (a) => { ASSETS = a; }, DUR: () => DUR };
`)(...names.map(n => env[n]));

const tick = () => new Promise(r => setTimeout(r, 0));

(async () => {
  /* ---- 1. a clip's tile is an image, and no decoder is ever opened ---- */
  MADE.length = 0;
  const thumb = api.videoThumb("a/shot.mp4");
  ck("the thumbnail is an <img>", thumb._tag, "img");
  ck("NO <video> IS CREATED", MADE.filter(e => e._tag === "video").length, 0);
  ck("it points at the server's thumbnail route", thumb.src,
     "/h3_toolkit/thumb?path=a%2Fshot.mp4&type=input&edge=192");
  ck("and leaves the loading to the browser", thumb.loading, "lazy");

  const outThumb = api.videoThumb("H3/take.mp4", "vthumb", "output");
  ck("the bin's rows read the OUTPUT folder", outThumb.src,
     "/h3_toolkit/thumb?path=H3%2Ftake.mp4&type=output&edge=192");

  /* ---- 2. a clip that cannot be decoded is not a broken gallery ------ */
  const t3 = api.videoThumb("gone.mp4");
  t3._fire("error");
  ck("a clip with no readable frame says so", t3._replacedBy.textContent, "!");

  /* ---- 3. the length comes from the server, not from the file -------- */
  MADE.length = 0;
  served["/h3_toolkit/clipinfo?path=late.mp4&type=output"] =
    { ok: true, frames: 60, seconds: 2.5 };
  const got = await api.learnDuration("late.mp4");
  ck("the length came back in frames", got, 60);
  ck("and was remembered", api.DUR()["late.mp4"], 60);
  ck("WITHOUT OPENING THE CLIP",
     MADE.filter(e => e._tag === "video").length, 0);
  // a second ask must not go back to the server
  served["/h3_toolkit/clipinfo?path=late.mp4&type=output"] = null;
  ck("a known clip costs nothing", await api.learnDuration("late.mp4"), 60);

  /* ---- 4. selecting does not rebuild the gallery --------------------- */
  const assets = ["a.png", "b.png", "c.mp4", "d.wav"];
  api.setAssets(assets);
  api.$("pick_q").value = "";
  api.$("pick_folder").value = "";
  api.$("pick_kind").value = "";
  const host = api.$("pick_tiles");
  api.setPick({ multi: true, chosen: new Set(), done: null, assets: null });
  MADE.length = 0;
  api.drawPicker();
  ck("a tile per asset", host._kids.length, 4);
  ck("still no decoder, even with a clip in the list",
     MADE.filter(e => e._tag === "video").length, 0);
  const first = host._kids[0];
  first.onclick();
  ck("the click selected it", [...api.getPick().chosen], ["a.png"]);
  ck("and marked it", first.classList.contains("sel"), true);
  ck("THE GALLERY WAS NOT REBUILT", host._kids.length, 4);
  ck("the same element, not a replacement", host._kids[0] === first, true);
  first.onclick();
  ck("clicking again deselects", [...api.getPick().chosen], []);

  api.setPick({ multi: false, chosen: new Set(), done: null, assets: null });
  api.drawPicker();
  host._kids[0].onclick();
  host._kids[1].onclick();
  ck("single-pick replaces the choice", [...api.getPick().chosen], ["b.png"]);
  ck("and clears the old mark", host._kids[0].classList.contains("sel"), false);

  /* ---- 5. the cap is announced, not silent -------------------------- */
  api.setAssets(Array.from({ length: 450 }, (_, i) => `f${i}.png`));
  api.setPick({ multi: true, chosen: new Set(), done: null, assets: null });
  api.drawPicker();
  ck("400 tiles plus one notice", host._kids.length, 401);
  ck("the notice counts the rest",
     host._kids[400].textContent.startsWith("50 more"), true);

  /* ---- 6. and the source says what it must not do ------------------- */
  // A FLAT GUARD, because this regressed once already: the fix was applied,
  // the symptom stayed, and the reason was that the tile still fetched the
  // clip. Nothing in the gallery path may construct a video element.
  const gallery = [
    src.match(/function videoThumb\([\s\S]*?\n\}/)[0],
    src.match(/function drawPicker\(\)[\s\S]*?\nfunction paintPickCount\([\s\S]*?\n\}/)[0],
    src.match(/function drawBin\(\)[\s\S]*?\n\}/)[0],
    src.match(/function learnDuration\([\s\S]*?\n\}/)[0],
  ].join("\n");
  ck("no createElement('video') anywhere in the gallery path",
     /createElement\(\s*["']video["']\s*\)/.test(gallery), false);
  ck("and no preload=metadata either",
     /preload\s*[=:]\s*["']metadata["']/.test(gallery), false);

  console.log();
  if (bad) { console.log(`FAIL — ${bad}`); process.exit(1); }
  console.log("galleries: server-rendered frames, no decoders, no rebuild");
})();
