// optionsFor against BOTH /object_info shapes.
//
// CJ: "Additional Refs gallery is still not working right. It says 'no clips
// in ComfyUI/input' and I know for a fact that is simply not true." 459 clips
// were sitting there.
//
// A legacy INPUT_TYPES combo serialises as [[...options], {}]; a V3
// io.Combo.Input as ["COMBO", { options: [...] }]. optionsFor read only the
// first, so every V3 node came back EMPTY AND SILENT — LoadVideo gave no
// clips, LoadAudio no voices, and ResolutionSelector no aspect ratios, which
// is why the hardcoded fallback list I should never have written looked like
// it was working.
const fs = require("fs");
const src = fs.readFileSync(__dirname + "/ui/index.html", "utf8")
  .match(/<script>([\s\S]*?)<\/script>/)[1];
const body = src.match(/async function optionsFor\([\s\S]*?\n\}/)[0];
let served = {};
const getJSON = async (u) => served[u];
const optionsFor = new Function("getJSON", `${body}; return optionsFor;`)(getJSON);
let bad = 0;
const ck = (l, g, w) => { const o = JSON.stringify(g) === JSON.stringify(w);
  console.log(`  ${o ? "ok  " : "FAIL"} ${l}: ${JSON.stringify(g)}`); if (!o) bad++; };
(async () => {
  // legacy INPUT_TYPES, as LoadImage serialises
  served["/object_info/LoadImage"] = { LoadImage: { input: { required: {
    image: [["a.png", "b.png"], { image_upload: true }] } } } };
  ck("legacy combo", await optionsFor("LoadImage", "image"), ["a.png", "b.png"]);
  // V3 io.Combo.Input, as LoadVideo serialises
  served["/object_info/LoadVideo"] = { LoadVideo: { input: { required: {
    file: ["COMBO", { options: ["x.mp4", "y.mp4"], upload: "video" }] } } } };
  ck("V3 combo", await optionsFor("LoadVideo", "file"), ["x.mp4", "y.mp4"]);
  // optional-section inputs, and a missing field
  served["/object_info/N"] = { N: { input: { optional: {
    k: ["COMBO", { options: ["q"] }] } } } };
  ck("an optional V3 combo", await optionsFor("N", "k"), ["q"]);
  ck("a field that is not there", await optionsFor("N", "nope"), []);
  served["/object_info/M"] = { M: { input: { required: { k: ["INT", {}] } } } };
  ck("a non-combo yields nothing", await optionsFor("M", "k"), []);
  console.log();
  if (bad) { console.log(`FAIL — ${bad}`); process.exit(1); }
  console.log("optionsFor: both /object_info shapes");
})();
