// H3 Audio Prompt — show only the fields the current mode actually reads, apply
// presets into the widgets, and draw the pacing plan on the node.
//
// The node has six text fields and three of them are ignored depending on mode,
// which is not guessable from the canvas. Rather than split into three nodes we
// hide what does not apply, so the node always shows exactly its live inputs.
import { app } from "../../scripts/app.js";

// which widgets each mode actually reads (see nodes.py)
const USED = {
  song:         ["style", "instrumentation", "voice", "room", "script"],
  speech:       ["voice", "room", "script"],
  instrumental: ["style", "instrumentation", "room"],
};
const ALL = ["style", "instrumentation", "voice", "room", "script"];

function setShown(node, name, shown) {
  const w = node.widgets?.find((x) => x.name === name);
  if (!w) return;
  if (!shown) {
    if (w.origType === undefined) w.origType = w.type;
    if (w.origComputeSize === undefined) w.origComputeSize = w.computeSize;
    w.type = "h3hidden";
    w.computeSize = () => [0, -4];
    if (w.element) w.element.style.display = "none";   // multiline DOM widgets
  } else if (w.origType !== undefined) {
    w.type = w.origType;
    w.computeSize = w.origComputeSize;
    if (w.element) w.element.style.display = "";
  }
  w.hidden = !shown;
}

function applyMode(node) {
  const mode = node.widgets?.find((w) => w.name === "mode")?.value ?? "song";
  const used = USED[mode] ?? ALL;
  for (const name of ALL) setShown(node, name, used.includes(name));
  // `voice` is replaced wholesale when the voice comes from ref_audio_1
  const fromAudio = node.widgets?.find((w) => w.name === "voices_from_audio")?.value;
  if (fromAudio && used.includes("voice")) setShown(node, "voice", false);
  // auto_fit_duration overrides seconds, so stop showing a number that is ignored
  const autofit = node.widgets?.find((w) => w.name === "auto_fit_duration")?.value;
  setShown(node, "seconds", !(autofit && mode !== "instrumental"));

  node.setSize(node.computeSize());
  app.graph.setDirtyCanvas(true, true);
}

// presets are also applied server-side; doing it here too means the fields fill
// in visibly so they can be edited rather than overridden invisibly
const PRESET_FIELDS = ["mode", "style", "instrumentation", "voice", "room", "script"];

async function applyPreset(node) {
  const w = node.widgets?.find((x) => x.name === "preset");
  if (!w || !w.value || w.value.startsWith("custom")) return;
  try {
    const r = await fetch("/h3_audio/preset?name=" + encodeURIComponent(w.value));
    if (!r.ok) return;
    const data = await r.json();
    for (const f of PRESET_FIELDS) {
      if (data[f] === undefined) continue;
      const t = node.widgets.find((x) => x.name === f);
      if (t) t.value = data[f];
    }
    w.value = "custom (use fields below)";   // so edits are not overwritten next run
    applyMode(node);
  } catch (e) { /* preset endpoint unavailable — server-side fallback still works */ }
}

// the lint report is only useful if you can see it without wiring an output
// character load/save report, same treatment as the lint node
app.registerExtension({
  name: "h3.character",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["H3Character", "H3CharacterSave", "H3Assemble", "H3AudioSlice",
          "H3Take", "H3Resolution"].includes(nodeData.name)) return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const text = message?.h3char?.[0];
      if (text === undefined) return;
      let w = this.widgets?.find((x) => x.name === "__info");
      if (!w) {
        w = this.addWidget("text", "__info", "", () => {}, { multiline: false });
        if (w.inputEl) w.inputEl.readOnly = true;
        w.serialize = false;
      }
      w.value = text;
      app.graph.setDirtyCanvas(true, true);
    };
  },
});

app.registerExtension({
  name: "h3.prompt.lint",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["H3PromptLint", "H3ScenePrompt", "H3LongFormLinks", "H3RewriterParse"].includes(nodeData.name)) return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const text = message?.h3lint?.[0];
      if (text === undefined) return;
      let w = this.widgets?.find((x) => x.name === "__report");
      if (!w) {
        w = this.addWidget("text", "__report", "", () => {}, { multiline: true });
        if (w.inputEl) w.inputEl.readOnly = true;
        w.serialize = false;
      }
      w.value = text;
      // colour the node by worst severity so a bad prompt is obvious at a glance
      this.color = text.startsWith("clean") ? "#233" :
                   /^\s*0 error/.test(text) ? "#432" : "#533";
      this.setSize(this.computeSize());
      app.graph.setDirtyCanvas(true, true);
    };
  },
});

app.registerExtension({
  name: "h3.audio.prompt",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["H3AudioPrompt", "H3ScenePrompt", "H3LongFormLinks"].includes(nodeData.name)) return;

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      for (const name of ["mode", "voices_from_audio", "auto_fit_duration"]) {
        const w = this.widgets?.find((x) => x.name === name);
        if (!w) continue;
        const prev = w.callback;
        w.callback = function () { const r = prev?.apply(this, arguments); applyMode(node); return r; };
      }
      const p = this.widgets?.find((x) => x.name === "preset");
      if (p) {
        const prev = p.callback;
        p.callback = function () { const r = prev?.apply(this, arguments); applyPreset(node); return r; };
      }
      const node = this;
      requestAnimationFrame(() => applyMode(node));
    };

    // draw the plan (line times, pauses, pacing warnings) on the node itself
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const text = message?.h3plan?.[0];
      if (text === undefined) return;
      let w = this.widgets?.find((x) => x.name === "__plan");
      if (!w) {
        w = this.addWidget("text", "__plan", "", () => {}, { multiline: true });
        w.inputEl && (w.inputEl.readOnly = true);
        w.serialize = false;
      }
      w.value = text;
      this.setSize(this.computeSize());
      app.graph.setDirtyCanvas(true, true);
    };
  },
});
