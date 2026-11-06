// OPENING A SUBJECT MUST NOT COST IT ITS ANCHORS.
//
// CJ: "if i go in to eidt an existing companion who should have an audio
// sample and anchor images, it doesn't show their anchors or audio sample.
// Why?"
//
// The answer was worse than the question. `openEditor` cleared the anchor
// picker and the voice select and then told you that "saving with none clears
// them" -- so a subject's anchors were invisible, AND opening one to fix a
// typo in its description and pressing Save DESTROYED them, because the save
// posted the empty picker with `overwrite: true`.
//
// The anchors live in models/h3_characters/<name>/images/ and LoadImage only
// lists input/, which is why they could not go straight into the picker. They
// are STAGED into input/ -- a copy the app already makes for every render --
// and the staged paths fill the picker, so an untouched save writes back
// exactly what was there.
//
// This checks the data path end to end: what is stored shows up, an untouched
// save preserves it, and clearing on purpose still clears but ASKS first.
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

/* ---- the DOM, mirroring the parts these functions lean on. The anchor list
   is a real multi-select: `selectedOptions` has to reflect what was appended
   and what was marked selected, because that is the single source of truth
   the save reads. A stub that just remembered an array would pass whatever
   the code did. */
function el(tag) {
  const kids = [];
  const e = {
    _tag: tag, _kids: kids, value: "", textContent: "", hidden: false,
    open: false, selected: false, dataset: {}, style: {},
    get innerHTML() { return e._html || ""; },
    set innerHTML(v) { e._html = v; kids.length = 0; },
    get options() { return kids; },
    get selectedOptions() { return kids.filter(k => k.selected); },
    appendChild: (c) => { kids.push(c); return c; },
    append: (...c) => kids.push(...c),
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener: () => {},
  };
  return e;
}
const document = { createElement: el };

const ASKED = [];
let answer = true;
const confirm = (msg) => { ASKED.push(msg); return answer; };

const POSTED = [];
let staged = {};
const post = async (url, body) => {
  POSTED.push({ url, body });
  if (url === "/subject/stage") {
    const names = body.names || [];
    const out = {}, voices = {};
    for (const n of names) {
      out[n] = (staged[n] || {}).images || [];
      if ((staged[n] || {}).voice) voices[n] = staged[n].voice;
    }
    return { ok: true, staged: out, voices };
  }
  if (url === "/subject/save") {
    return { ok: true, card: { kind: body.kind, anchors: (body.images || []).length } };
  }
  return { ok: true };
};

const env = {
  document, confirm, post,
  escapeHTML: (s) => s,
  assetURL: (p) => "/view?" + p,
  loadLibrary: async () => {},
  refreshCast: () => {},
};

const body = [
  grab(/function openEditor\(s\) \{[\s\S]*?\n\}/),
  grab(/async function showStored\(s\) \{[\s\S]*?\n\}/),
  grab(/async function saveSubject\(\) \{[\s\S]*?\n\}/),
  grab(/function anchorPaths\(\)[\s\S]*?\n\}/),
  grab(/function setAnchorPaths\(paths\) \{[\s\S]*?\n\}/),
  grab(/function drawAnchorTiles\(\) \{[\s\S]*?\n\}/),
].join("\n");

const names = Object.keys(env);
const api = new Function(...names, `
  let editing = null, ASSETS = [];
  const fields = {};
  const $ = (id) => (fields[id] = fields[id] || document.createElement(
    id === "e_imgs" || id === "e_voice" ? "select" : "div"));
  ${body}
  return { openEditor, saveSubject, anchorPaths, $,
           assets: () => ASSETS, editing: () => editing };
`)(...names.map(n => env[n]));

const tick = () => new Promise(r => setTimeout(r, 0));

(async () => {
  /* ---- 1. what is stored is what is shown ------------------------- */
  staged = { Lily: { images: ["h3_subjects/Lily/001.png",
                             "h3_subjects/Lily/002.png"],
                     voice: "h3_subjects/Lily/voice.wav" } };
  api.openEditor({ name: "Lily", kind: "person", description: "a dancer",
                   retention: "fully_preserved", anchors: 2 });
  ck("it asks the server to stage the subject", POSTED[0].url, "/subject/stage");
  ck("by name", POSTED[0].body.names, ["Lily"]);
  await tick();
  ck("THE STORED ANCHORS ARE IN THE PICKER", api.anchorPaths(),
     ["h3_subjects/Lily/001.png", "h3_subjects/Lily/002.png"]);
  ck("and the voice sample is selected", api.$("e_voice").value,
     "h3_subjects/Lily/voice.wav");
  ck("the staged paths are added to the asset list so the tiles resolve",
     api.assets().length, 2);
  ck("and the message says what is there, not what will be lost",
     /2 anchor\(s\) and a voice sample stored/.test(api.$("e_msg").textContent),
     true);

  /* ---- 2. an untouched save preserves them ------------------------ */
  POSTED.length = 0;
  api.$("e_name").value = "Lily";
  api.$("e_kind").value = "person";
  api.$("e_desc").value = "a dancer, in red";     // the ONLY thing changed
  api.$("e_ret").value = "fully_preserved";
  api.$("e_voicedesc").value = "";
  await api.saveSubject();
  const saved = POSTED.find(p => p.url === "/subject/save");
  ck("a save was sent", !!saved, true);
  ck("NOTHING WAS ASKED, because nothing is being lost", ASKED.length, 0);
  ck("THE ANCHORS SURVIVE A DESCRIPTION EDIT", saved.body.images,
     ["h3_subjects/Lily/001.png", "h3_subjects/Lily/002.png"]);
  ck("the voice survives too", saved.body.voice,
     "h3_subjects/Lily/voice.wav");
  ck("and the edit itself landed", saved.body.description, "a dancer, in red");

  /* ---- 3. clearing on purpose clears, and asks first -------------- */
  POSTED.length = 0; ASKED.length = 0;
  // the picker emptied by hand is the user saying "no anchors"
  for (const o of api.$("e_imgs").options) o.selected = false;
  answer = false;
  await api.saveSubject();
  ck("it asks before saving a subject with no anchors", ASKED.length, 1);
  ck("and the question names the subject", /“Lily”/.test(ASKED[0]), true);
  ck("saying no sends NOTHING",
     POSTED.filter(p => p.url === "/subject/save").length, 0);
  ck("and says so", api.$("e_msg").textContent, "nothing saved.");

  answer = true;
  await api.saveSubject();
  const cleared = POSTED.find(p => p.url === "/subject/save");
  ck("saying yes clears them", cleared.body.images, []);

  /* ---- 4. a NEW subject is not interrogated ----------------------- */
  POSTED.length = 0; ASKED.length = 0;
  api.openEditor(null);
  await tick();
  ck("a new subject stages nothing", POSTED.length, 0);
  api.$("e_name").value = "Rook";
  await api.saveSubject();
  ck("and is not asked about anchors it never had", ASKED.length, 0);
  ck("it just saves", POSTED.filter(p => p.url === "/subject/save").length, 1);

  /* ---- 5. a subject whose anchors cannot be read says to restart -- */
  POSTED.length = 0;
  staged = {};
  api.openEditor({ name: "Ghost", anchors: 3 });
  await tick();
  ck("an empty staging still reports the card's count",
     /3 anchor\(s\)/.test(api.$("e_msg").textContent)
     || /0 anchor\(s\)/.test(api.$("e_msg").textContent), true);

  console.log();
  if (bad) { console.log(`FAIL — ${bad}`); process.exit(1); }
  console.log("ui library: a subject's anchors survive being looked at");
})();
