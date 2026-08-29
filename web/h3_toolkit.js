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

// Grow to fit, never shrink past what the user set.
//
// `computeSize()` is the node's MINIMUM, so `setSize(computeSize())` throws away
// any manual resize. These nodes ran it on every execution -- and H3LongFormLinks
// is registered by two extensions, so it collapsed twice per run -- which meant
// the prompt fields you had just dragged open shut again the moment you queued.
function fitAtLeast(node) {
  const min = node.computeSize();
  node.setSize([Math.max(node.size[0], min[0]), Math.max(node.size[1], min[1])]);
}

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

  fitAtLeast(node);
  app.graph.setDirtyCanvas(true, true);
}

// presets are also applied server-side; doing it here too means the fields fill
// in visibly so they can be edited rather than overridden invisibly
const PRESET_FIELDS = ["mode", "style", "instrumentation", "voice", "room", "script"];

async function applyPreset(node) {
  const w = node.widgets?.find((x) => x.name === "preset");
  if (!w || !w.value || w.value.startsWith("custom")) return;
  try {
    const r = await fetch("/h3_toolkit/preset?name=" + encodeURIComponent(w.value));
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
          "H3Take", "H3Resolution", "H3Canvas"].includes(nodeData.name)) return;
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
      // colour the node by worst severity so a bad prompt is obvious at a glance
      const clean = text.startsWith("clean");
      this.color = clean ? "#233" : /^\s*0 error/.test(text) ? "#432" : "#533";
      let w = this.widgets?.find((x) => x.name === "__report");
      if (clean) {
        // nothing to say, and the colour already says it -- give the height back
        // to the fields you are actually typing in
        if (w) {
          w.inputEl?.remove();
          this.widgets.splice(this.widgets.indexOf(w), 1);
        }
        app.graph.setDirtyCanvas(true, true);
        return;
      }
      if (!w) {
        w = this.addWidget("text", "__report", "", () => {}, { multiline: true });
        if (w.inputEl) w.inputEl.readOnly = true;
        w.serialize = false;
      }
      w.value = text;
      fitAtLeast(this);
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
      // No forced field heights. A minHeight on the text areas reserved space
      // whether or not the text needed it, so a one-line `head` sat above a
      // 140px gap. fitAtLeast() already stops the node collapsing on every run,
      // which is what was actually wrong — the size you drag it to is now the
      // size it keeps, and the fields fill it.
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
      fitAtLeast(this);
      app.graph.setDirtyCanvas(true, true);
    };
  },
});

// A widget that is still being read is editable; one that has been overridden by
// a wire should look overridden. `seconds_per_link` cannot be removed from the
// node -- widgets_values is positional and deleting it would shift `seed` into
// its slot in every saved graph -- so when `chunk_frames` is connected it is
// greyed and relabelled instead, and the label says what took over.
app.registerExtension({
  name: "h3.longform.override",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "H3LongFormLinks") return;

    // Only `seconds_per_link`. It has an OVERRIDE -- a wired chunk_frames
    // supersedes it, and the node ignores it -- so showing it live is a lie.
    //
    // `seed` has no override. It is unwired in the chunked example workflows,
    // which is not the same thing: feeding it to the sampler's noise_seed works
    // under chunking exactly as it does in the chain, and holds one seed across
    // every chunk. Greying it would be greying a working connection.
    const OVERRIDDEN = { seconds_per_link: "seconds_per_link (chunk length wins)" };

    function apply(node) {
      // The node reads the VALUE, not the wire: `int(chunk_frames or 0) > 0` is
      // what makes seconds_per_link inert, so a typed 192 counts exactly as much
      // as a connection. Checking only for a link left a typed value looking
      // live and sent me chasing a wire that was never needed.
      const w = node.widgets?.find((x) => x.name === "chunk_frames");
      const wired = (node.inputs || []).some(
          (i) => i.name === "chunk_frames" && i.link != null)
        || Number(w?.value) > 0;
      for (const [name, label] of Object.entries(OVERRIDDEN)) {
        const w = node.widgets?.find((x) => x.name === name);
        if (!w) continue;
        if (w.h3label === undefined) w.h3label = w.label ?? w.name;
        w.disabled = wired;
        w.label = wired ? label : w.h3label;
      }
      app.graph?.setDirtyCanvas(true, true);
    }

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const node = this;
      requestAnimationFrame(() => apply(node));
    };

    // a saved graph restores its links after construction, so re-check then
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      configure?.apply(this, arguments);
      const node = this;
      requestAnimationFrame(() => apply(node));
    };

    const created2 = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created2?.apply(this, arguments);
      const node = this;
      const w = node.widgets?.find((x) => x.name === "chunk_frames");
      if (w) {
        const prev = w.callback;
        w.callback = function () {
          const r = prev?.apply(this, arguments);
          apply(node);
          return r;
        };
      }
    };

    const changed = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      changed?.apply(this, arguments);
      apply(this);
    };
  },
});
