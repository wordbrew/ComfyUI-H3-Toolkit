// The timeline, driven through a stubbed DOM.
//
// WHAT THIS IS DEFENDING
//   The drag arithmetic and the write-back. Both are silent when wrong: a
//   handle that moves the start AND the end looks fine while you drag it and
//   emits a different clip; a region drag that squashes instead of stopping at
//   the edge quietly changes the frame count you spent the session choosing.
//
//   And the write-back is the contract between this file and frames.py. They
//   parse the same `start count [label]` grammar, so a change to one that the
//   other does not follow means the timeline draws something the node does not
//   render -- which is the entire class of bug this node was built to end.
//
//     node test_videotrim_panel.js

const fs = require("fs");
const path = require("path");

// ------------------------------------------------------------- a fake DOM --
class Node_ {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.style = { cssText: "" };
    this.listeners = {};
    this.value = "";
    this.textContent = "";
    this.options = [];
    this.selectedIndex = 0;
    this._rect = { left: 0, width: 400, top: 0, height: 44 };
    this.scrollHeight = 0;
  }
  append(...k) { for (const c of k) { c.parentNode = this; this.children.push(c); } }
  replaceChildren(...k) { this.children = []; this.append(...k); }
  addEventListener(t, fn) { (this.listeners[t] ||= []).push(fn); }
  removeEventListener(t, fn) {
    this.listeners[t] = (this.listeners[t] || []).filter((f) => f !== fn);
  }
  fire(t, ev = {}) {
    const e = { preventDefault() {}, stopPropagation() {}, target: this, ...ev };
    for (const fn of this.listeners[t] || []) fn(e);
    // a real element dispatches to the `on<type>` PROPERTY as well, and the
    // panel uses both -- listeners for the drag handles, properties for the
    // buttons. A stub that only knows one silently skips half the controls.
    const prop = this["on" + t];
    if (typeof prop === "function") prop.call(this, e);
  }
  getBoundingClientRect() { return this._rect; }
  querySelector(sel) {
    const want = sel.toUpperCase();
    const walk = (n) => {
      for (const c of n.children) {
        if ((c.tagName || "").toUpperCase() === want) return c;
        const hit = walk(c);
        if (hit) return hit;
      }
      return null;
    };
    return walk(this);
  }
  click() { this.onclick?.({ stopPropagation() {}, preventDefault() {} }); }
  setAttribute(k, v) { this[k] = v; }
  // a BOOLEAN attribute reflects as false when removed, not as undefined --
  // `video.controls` after removeAttribute("controls") is false in a browser
  removeAttribute(k) { this[k] = false; }
  focus() { global.document.activeElement = this; }
  get firstChild() { return this.children[0]; }
  play() { this.paused = false; return Promise.resolve(); }
  pause() { this.paused = true; }
}

function installDom() {
  const doc = {
    createElement: (tag) => new Node_(tag),
    activeElement: null,
  };
  global.document = doc;
  global.HTMLInputElement = Node_;      // instanceof checks in the key handler
  global.HTMLSelectElement = Node_;
  global.window = {
    listeners: {},
    addEventListener(t, fn) { (this.listeners[t] ||= []).push(fn); },
    removeEventListener(t, fn) {
      this.listeners[t] = (this.listeners[t] || []).filter((f) => f !== fn);
    },
    fire(t, ev) { for (const fn of this.listeners[t] || []) fn(ev); },
  };
  global.FormData = class { constructor() { this.parts = {}; }
                            append(k, v) { this.parts[k] = v; } };
  // tick() would spin forever, so hold the callback and let a test pump it by
  // hand. The height watcher lives in there and has to be exercised, so a
  // no-op rAF would leave it untested -- which is what it was.
  global.__frame = null;
  global.requestAnimationFrame = (fn) => { global.__frame = fn; return 1; };
  global.cancelAnimationFrame = () => { global.__frame = null; };
  global.fetch = async () => ({
    json: async () => ({ ok: true, frames: 300, fps: 30, width: 1920,
                         height: 1080, duration: 10 }),
  });
}

// ------------------------------------------------------------ the module --
function loadModule() {
  const src = fs.readFileSync(path.join(__dirname, "web/videotrim.js"), "utf8")
    .replace(/^import \{ app \} from .*$/m, "const app = __app;");
  // capture the extension, so a test can run the real registration path
  const ext = {};
  const factory = new Function("__app", src + "\n;return { attach, parseSpans, formatSpans, clampSpan, refitSpan, hideStorageWidgets, suppressForeignPlayers, panelHeight };");
  const M = factory({ registerExtension(e) { Object.assign(ext, e); } });
  M.ext = ext;
  return M;
}

// --------------------------------------------------------------- a fake node
function fakeNode(overrides = {}) {
  // every real widget has a type; hideStorageWidgets remembers it so the panel
  // can be removed without stranding the node
  const w = (name, value) => ({ name, value, callback: null,
                                type: typeof value === "number" ? "number"
                                                                : "text" });
  return {
    // a real combo widget carries its choices in options.values, which is what
    // a freshly uploaded file has to be added to before it can be selected
    widgets: [{ ...w("file", "clip.mp4"), type: "combo",
                options: { values: ["clip.mp4"] } },
              w("spans", "0 0"), w("span_index", 0),
              w("force_rate", 0), w("select_every_nth", 1),
              ...(overrides.widgets || [])],
    size: [400, 300],
    setDirtyCanvas() {},
    setSize(v) { this.size = v; },
    // THE REAL ONE ASKS EVERY WIDGET. A fake that returns a constant never
    // calls the panel's own computeSize -- which is exactly how a widget
    // computeSize that threw ReferenceError got past a green suite.
    computeSize() {
      let h = 120;                       // the node's own widget rows
      for (const w of this.widgets || []) {
        if (typeof w.computeSize === "function") h += w.computeSize(this.size[0])[1];
      }
      // AND ITS OWN NATURAL WIDTH, which is what litegraph returns -- derived
      // from the widest widget, not from the size the node currently has. A
      // fake that echoed size[0] back hid the fact that
      // `setSize(this.computeSize())` throws the chosen width away.
      return [210, h];
    },
    // the real one returns the widget so the extension can set its size hooks
    addDOMWidget(name, type, element) {
      const w = { name, type, element, options: {} };
      this.widgets.push(w);
      this.__domWidget = w;
      return w;
    },
  };
}

const get = (node, n) => node.widgets.find((x) => x.name === n);

// ------------------------------------------------------------------ checks
let fails = 0;
function check(label, got, want) {
  const a = JSON.stringify(got), b = JSON.stringify(want);
  if (a !== b) { fails++; console.log(`  FAIL ${label}: got ${a}, want ${b}`); }
  else console.log(`  ok   ${label}`);
}
function ok(label, cond) { check(label, !!cond, true); }

// find a node in the tree by the cursor its style declares — the handles have
// no ids, and looking them up by role is what a user does too
function findAll(root, pred, out = []) {
  if (pred(root)) out.push(root);
  for (const c of root.children) findAll(c, pred, out);
  return out;
}

installDom();
const M = loadModule();

console.log("the grammar matches frames.py, because one file writes what the "
            + "other reads");
check("a span line parses",
      M.parseSpans("10 50 hello", 300), [{ start: 10, count: 50, label: "hello" }]);
check("count 0 means to the end",
      M.parseSpans("10 0", 300), [{ start: 10, count: 290, label: "" }]);
check("nothing at all covers the clip",
      M.parseSpans("", 300), [{ start: 0, count: 300, label: "" }]);
check("and it round-trips",
      M.formatSpans(M.parseSpans("0 192 opening\n240 60 turn", 300)),
      "0 192 opening\n240 60 turn");
// CUT, NEVER SHIFTED -- a span of the right length from the wrong place is the
// worse failure, because it looks correct
check("a span past the end is cut, not moved",
      M.clampSpan({ start: 280, count: 50 }, 300), { start: 280, count: 20 });

console.log("the ruler is in OUTPUT frames, so force_rate changes its length");
async function panel(widgets) {
  const node = fakeNode();
  for (const [n, v] of Object.entries(widgets || {})) get(node, n).value = v;
  const root = M.attach(node);
  await new Promise((r) => setTimeout(r, 0));   // let the probe resolve
  return { node, root };
}

(async () => {
  // 300 source frames @ 30fps
  let { node } = await panel({ spans: "0 0" });
  check("no force_rate keeps the source's 300", get(node, "spans").value, "0 300");

  ({ node } = await panel({ spans: "0 0", force_rate: 24 }));
  // 300 frames @30 is 10s; 10s @24 is 240
  check("force_rate 24 makes it 240", get(node, "spans").value, "0 240");

  ({ node } = await panel({ spans: "0 0", force_rate: 24, select_every_nth: 2 }));
  check("and every 2nd makes it 120", get(node, "spans").value, "0 120");

  console.log("dragging: the ends resize, the middle SLIDES");
  const drag = async (kind, fromPx, toPx, spans, extra) => {
    const { node, root } = await panel({ spans, ...(extra || {}) });
    const bar = findAll(root, (n) => n.style.cssText.includes("height:44px"))[0];
    const handles = findAll(bar, (n) => n.style.cssText.includes("cursor:ew-resize"));
    const region = findAll(bar, (n) => n.style.cssText.includes("cursor:grab"))[0];
    const video = findAll(root, (n) => n.tagName === "video")[0];
    video.paused = true;
    const target = kind === "in" ? handles[0]
                 : kind === "out" ? handles[1] : region;
    target.fire("pointerdown", { clientX: fromPx });
    global.window.fire("pointermove", { clientX: toPx, buttons: 1 });
    global.window.fire("pointerup", {});
    return { spans: get(node, "spans").value, video, node, root };
  };
  const grab = async (...a) => (await drag(...a)).spans;

  // the bar is 400px wide over 300 frames, so 100px = 75 frames
  check("the whole block slides and keeps its width",
        await grab("region", 0, 100, "0 100"), "75 100");
  check("the in handle moves the start and the END STAYS PUT",
        await grab("in", 0, 100, "0 200"), "75 125");
  check("the out handle changes only the count",
        await grab("out", 0, -100, "0 200"), "0 125");

  console.log("and it stops at the edges rather than squashing");
  // A REGION DRAGGED PAST THE END must keep its length: silently shortening
  // the clip you spent the session choosing is the failure worth guarding
  check("sliding past the end pins the block, whole",
        await grab("region", 0, 9999, "0 100"), "200 100");
  check("sliding before the start pins it too",
        await grab("region", 0, -9999, "100 100"), "0 100");
  check("the in handle cannot cross the out handle",
        await grab("in", 0, 9999, "0 50"), "49 1");
  check("the out handle cannot cross the in handle",
        await grab("out", 0, -9999, "100 50"), "100 1");

  console.log("several spans, and span_index follows the one you are editing");
  {
    const { node, root } = await panel({ spans: "0 100 one" });
    const btns = findAll(root, (n) => n.tagName === "button");
    const add = btns.find((b) => b.textContent === "+ span");
    const del = btns.find((b) => b.textContent === "− span");
    add.fire("click");
    check("a new span follows the last and keeps its length",
          get(node, "spans").value, "0 100 one\n100 100");
    check("and becomes the active one", get(node, "span_index").value, 1);
    add.fire("click");
    check("a third", get(node, "spans").value, "0 100 one\n100 100\n200 100");
    del.fire("click");
    check("removing drops the active one",
          get(node, "spans").value, "0 100 one\n100 100");
    // three spans, the third removed -> you land on the second, index 1
    check("and steps back to the one before it",
          get(node, "span_index").value, 1);
    del.fire("click");
    del.fire("click");
    check("the last span cannot be removed",
          get(node, "spans").value.split("\n").length, 1);
  }

  console.log("the keys are on the PANEL, not on window");
  // ComfyUI owns space (pan the canvas) and the arrows (nudge a node). A global
  // listener here would break those everywhere else in the app.
  {
    const before = Object.keys(global.window.listeners).length;
    const { root } = await panel({ spans: "0 100" });
    ok("no keydown handler was added to window",
       !(global.window.listeners.keydown || []).length);
    ok("the panel takes keys itself", (root.listeners.keydown || []).length === 1);
    ok("and it is focusable, or it would never receive them", root.tabIndex === 0);
    check("window gained no listeners at rest",
          Object.keys(global.window.listeners).length, before);
  }

  console.log("the panel never lets a press reach the canvas");
  {
    // A POINTERDOWN THAT REACHES THE CANVAS DRAGS THE NODE. Every press in the
    // panel is the start of a scrub or a handle drag, so the node would slide
    // out from under the cursor mid-drag.
    const { root } = await panel({ spans: "0 100" });
    const downs = root.listeners.pointerdown || [];
    ok("the root swallows pointerdown", downs.length >= 1);
    let stopped = false;
    downs[0]({ stopPropagation: () => { stopped = true; }, preventDefault() {} });
    ok("and it stops propagation", stopped);
  }

  console.log("the video shows no native controls, which would sit over ours");
  {
    const { root } = await panel({ spans: "0 100" });
    const video = findAll(root, (n) => n.tagName === "video")[0];
    ok("a video is drawn", !!video);
    check("controls are off", video.controls, false);
    ok("and the attribute is gone too", !("controls" in video) ||
       video.controls === false);
  }

  console.log("the banner says which node you are looking at");
  {
    const node = fakeNode();
    const root = M.attach(node, false);
    await new Promise((r) => setTimeout(r, 0));
    // found by what it SAYS, not by where it sits -- an index breaks the moment
    // anything is added above it, which is exactly what happened
    const said = (r) => findAll(r, (n) => /^Emits /.test(n.textContent || ""))[0];
    ok("the single node says it emits ONE",
       /Emits ONE span: number 1 of 1/.test(said(root)?.textContent || ""));

    const node2 = fakeNode();
    const root2 = M.attach(node2, true);
    await new Promise((r) => setTimeout(r, 0));
    const b2 = said(root2)?.textContent || "";
    ok("the batch node says it emits ALL", /Emits ALL 1 span\./.test(b2));
    ok("and warns the graph below runs per span", /runs 1 time/.test(b2));
  }

  console.log("the storage widgets are hidden BY THE PANEL, not by the node");
  {
    const node = fakeNode();
    M.hideStorageWidgets(node);
    check("spans is hidden", get(node, "spans").type, "hidden");
    check("span_index too", get(node, "span_index").type, "hidden");
    check("it takes no row height", get(node, "spans").computeSize(), [0, -4]);
    // with this file gone they come back and the node is still usable by
    // typing `start count` into a text box -- the widgets stay the truth
    ok("and the original type is remembered",
       get(node, "spans").origType !== undefined);
    ok("the file widget is untouched", get(node, "file").type !== "hidden");
  }

  console.log("the storage widgets hide their ELEMENT, not just their row");
  {
    // A MULTILINE WIDGET IS A DOM ELEMENT. Setting `type` governs the canvas
    // row only -- the textarea stays positioned over the node, which is how a
    // "hidden" spans box ended up drawn on top of force_rate.
    const node = fakeNode();
    const spansW = get(node, "spans");
    spansW.element = new Node_("textarea");
    M.hideStorageWidgets(node);
    check("the row is hidden", spansW.type, "hidden");
    check("and so is the element", spansW.element.style.display, "none");
  }

  console.log("ComfyUI's own video preview is suppressed, so there is one player");
  {
    // found by SHAPE, not by name: whatever the frontend calls its preview
    // widget is not ours to depend on, but "a DOM widget holding a <video>
    // that is not the one we built" is stable
    const node = fakeNode();
    const mineEl = new Node_("div");
    mineEl.append(new Node_("video"));
    const mine = { name: "timeline", type: "videotrim", element: mineEl };
    const foreignEl = new Node_("div");
    foreignEl.append(new Node_("video"));
    const foreign = { name: "$$video-preview", type: "preview",
                      element: foreignEl };
    node.widgets.push(mine, foreign);

    M.suppressForeignPlayers(node, mine);
    check("the other player is hidden", foreign.type, "hidden");
    check("its element too", foreignEl.style.display, "none");
    ok("ours is untouched", mine.type === "videotrim");
    ok("and so are widgets with no video", get(node, "file").type !== "hidden");
  }

  console.log("it loops silently, and finds its voice when you hover");
  {
    // several of these on a canvas all talking at once is unusable; a silent
    // loop costs nothing to glance at. Muted is also the only autoplay a
    // browser allows without a gesture.
    const { root } = await panel({ spans: "0 100" });
    const video = findAll(root, (n) => n.tagName === "video")[0];
    check("it autoplays", video.autoplay, true);
    check("and loops", video.loop, true);
    check("silent at rest", video.muted, true);

    root.fire("pointerenter");
    check("hovering brings the sound up", video.muted, false);
    root.fire("pointerleave");
    check("and leaving takes it away", video.muted, true);
  }

  console.log("the speaker button is a LOCK on that, not a toggle of the moment");
  {
    const { root } = await panel({ spans: "0 100" });
    const video = findAll(root, (n) => n.tagName === "video")[0];
    const mute = findAll(root, (n) => n.tagName === "button")
      .find((b) => b.textContent === "🔊" || b.textContent === "🔇");
    ok("there is one", !!mute);
    root.fire("pointerenter");
    check("audio is allowed by default", video.muted, false);
    mute.fire("click");
    check("silencing it mutes even while hovering", video.muted, true);
    check("and the button says so", mute.textContent, "🔇");
    root.fire("pointerleave");
    root.fire("pointerenter");
    check("hover cannot override the lock", video.muted, true);
    mute.fire("click");
    check("unlocking restores it, still hovering", video.muted, false);
  }

  console.log("a blocked play() forces muted rather than doing nothing");
  {
    const { root } = await panel({ spans: "0 100" });
    const video = findAll(root, (n) => n.tagName === "video")[0];
    let tries = 0;
    video.play = () => {
      tries++;
      return tries === 1 ? Promise.reject(new Error("blocked"))
                         : Promise.resolve();
    };
    video.paused = true;
    const play = findAll(root, (n) => n.tagName === "button")
      .find((b) => b.textContent === "▶");
    play.fire("click");
    await new Promise((r) => setTimeout(r, 0));
    check("it tried twice", tries, 2);
    check("the second time muted", video.muted, true);
    ok("and it said why",
       !!findAll(root, (n) => (n.textContent || "").includes("blocked playback"))[0]);
  }

  console.log("a shorter clip keeps the LENGTH and slides, it does not eat it");
  {
    // THE STUCK-AT-1 BUG. clampSpan pins the start and cuts the count, so a
    // span starting past the end of a shorter clip became (last frame, 1) --
    // and that got written back, poisoning every clip after it.
    check("a span past the end of a short clip slides back",
          M.refitSpan({ start: 100, count: 120 }, 60), { start: 0, count: 60 });
    check("one that merely overruns keeps its start",
          M.refitSpan({ start: 10, count: 20 }, 60), { start: 10, count: 20 });
    check("one near the end slides just enough",
          M.refitSpan({ start: 50, count: 20 }, 60), { start: 40, count: 20 });
    check("count 0 still means the whole clip",
          M.refitSpan({ start: 0, count: 0 }, 60), { start: 0, count: 60 });
    // and the old rule, for contrast: this is what it used to produce
    check("clampSpan, the one-edit rule, still truncates",
          M.clampSpan({ start: 100, count: 120 }, 60), { start: 59, count: 1 });

    // end to end: a long clip, then a short one, then a long one again
    const { node } = await panel({ spans: "100 120" });   // probe says 300@30
    check("the long clip keeps it", get(node, "spans").value, "100 120");
    global.fetch = async () => ({
      json: async () => ({ ok: true, frames: 60, fps: 30, width: 640,
                           height: 480, duration: 2 }),
    });
    get(node, "file").callback("short.mp4");
    await new Promise((r) => setTimeout(r, 0));
    check("a 60-frame clip cuts it to 60, not to 1",
          get(node, "spans").value, "0 60");
    global.fetch = async () => ({
      json: async () => ({ ok: true, frames: 300, fps: 30, width: 640,
                           height: 480, duration: 10 }),
    });
    get(node, "file").callback("long.mp4");
    await new Promise((r) => setTimeout(r, 0));
    check("and the next clip is not stuck at 1",
          get(node, "spans").value, "0 60");
  }

  console.log("a clip that measures ONE frame never overwrites the spans");
  {
    // ComfyUI's get_frame_count() returns 1 for a whole class of file. The
    // server corrects it; this is the second line, because persisting a refit
    // against a bad total is what made the damage outlive the clip.
    const { node } = await panel({ spans: "0 120 keep" });
    global.fetch = async () => ({
      json: async () => ({ ok: true, frames: 1, fps: 30, width: 640,
                           height: 480, duration: 3 }),
    });
    get(node, "file").callback("weird.mp4");
    await new Promise((r) => setTimeout(r, 0));
    check("the span is left alone", get(node, "spans").value, "0 120 keep");
    global.fetch = async () => ({
      json: async () => ({ ok: true, frames: 300, fps: 30, width: 640,
                           height: 480, duration: 10 }),
    });
    get(node, "file").callback("fine.mp4");
    await new Promise((r) => setTimeout(r, 0));
    check("and survives to the next good clip",
          get(node, "spans").value, "0 120 keep");
  }

  console.log("and a failed probe never flattens the spans");
  {
    // total() is 0 when the probe fails; writing then would wipe every span --
    // the same data loss by a different route
    const { node } = await panel({ spans: "10 50 keep me" });
    global.fetch = async () => { throw new Error("no server"); };
    get(node, "file").callback("gone.mp4");
    await new Promise((r) => setTimeout(r, 0));
    check("the text is untouched", get(node, "spans").value, "10 50 keep me");
    global.fetch = async () => ({
      json: async () => ({ ok: true, frames: 300, fps: 30, width: 640,
                           height: 480, duration: 10 }),
    });
  }

  console.log("the panel takes the file itself, because it covers the node");
  {
    // A FILE DROPPED ON THE NODE LANDS ON THIS DIV and the canvas never sees
    // it -- which is why dropping on the node did nothing. The panel does the
    // upload rather than forwarding an event to a handler it does not own.
    const { node, root } = await panel({ spans: "0 100" });
    let posted = null;
    global.fetch = async (url, opts) => {
      if (url.startsWith("/upload/image")) {
        posted = opts.body.parts;
        return { ok: true, json: async () => ({ name: "dropped.mp4",
                                                subfolder: "", type: "input" }) };
      }
      return { json: async () => ({ ok: true, frames: 300, fps: 30,
                                    width: 640, height: 480, duration: 10 }) };
    };

    root.fire("drop", { dataTransfer: { files: [{ name: "dropped.mp4" }] } });
    await new Promise((r) => setTimeout(r, 0));
    ok("it posted the file", !!posted);
    check("as an upload to the input folder", posted.type, "input");
    check("under the key the route reads", !!posted.image, true);
    check("and the widget now points at it",
          get(node, "file").value, "dropped.mp4");
    ok("which the combo has heard of",
       (get(node, "file").options?.values || []).includes("dropped.mp4"));
  }

  console.log("and Choose video is a LABEL, so the picker opens with no JS");
  {
    // Forwarding a click to a hidden input depends on the gesture surviving
    // whatever the frontend does to events inside a DOM widget. A <label>
    // wrapping its own input opens the picker natively -- nothing in the path
    // to break. The structure IS the behaviour, so the structure is the check.
    const { root } = await panel({ spans: "0 100" });
    const open = findAll(root, (n) => n.tagName === "label")[0];
    ok("there is a control", !!open);
    check("and it is a label, not a button", open.tagName, "label");
    const picker = findAll(open, (n) => n.type === "file")[0];
    ok("with its file input INSIDE it, which is what makes it work", !!picker);
    ok("choosing a file is what triggers the upload",
       typeof picker.onchange === "function");
  }

  console.log("a failed upload says so rather than doing nothing");
  {
    const { root } = await panel({ spans: "0 100" });
    global.fetch = async (url) => {
      if (url.startsWith("/upload/image")) {
        return { ok: false, status: 413, statusText: "Payload Too Large" };
      }
      return { json: async () => ({ ok: true, frames: 300, fps: 30,
                                    width: 640, height: 480, duration: 10 }) };
    };
    root.fire("drop", { dataTransfer: { files: [{ name: "huge.mp4" }] } });
    await new Promise((r) => setTimeout(r, 0));
    ok("the reason is on the node",
       !!findAll(root, (n) => (n.textContent || "").includes("upload failed"))[0]);
    global.fetch = async () => ({
      json: async () => ({ ok: true, frames: 300, fps: 30, width: 640,
                           height: 480, duration: 10 }),
    });
  }

  console.log("dragging seeks the preview, so you are not choosing a frame blind");
  {
    // 400px over 300 frames at 30fps: 100px is 75 frames is 2.5s
    const a = await drag("in", 0, 100, "0 200");
    check("the in handle shows its new first frame", a.video.currentTime, 2.5);

    const b = await drag("region", 0, 100, "0 100");
    check("sliding the block shows where it now starts",
          b.video.currentTime, 2.5);

    // the OUT point is the last frame INSIDE the span, not the first one after
    // -- seeking to the span's end would show a frame the render will not hold
    const c = await drag("out", 0, -100, "0 200");
    const outFrame = 125;                       // 200 - 75
    check("the out handle shows the last frame it keeps",
          +c.video.currentTime.toFixed(4),
          +((outFrame / 30) - (1 / 30)).toFixed(4));
  }

  console.log("but it does not fight playback, which already loops in the span");
  {
    const { root } = await panel({ spans: "0 200" });
    const video = findAll(root, (n) => n.tagName === "video")[0];
    video.paused = false;
    video.currentTime = 4;
    const bar = findAll(root, (n) => n.style.cssText.includes("height:44px"))[0];
    const handles = findAll(bar, (n) =>
      n.style.cssText.includes("cursor:ew-resize"));
    handles[0].fire("pointerdown", { clientX: 0 });
    global.window.fire("pointermove", { clientX: 100, buttons: 1 });
    global.window.fire("pointerup", {});
    check("a playing video is left where it was", video.currentTime, 4);
  }

  console.log("and a drag that does not cross a frame is not a change at all");
  {
    // dozens of pointermoves per second land on the frame already set; without
    // this every one writes the widget and seeks the video
    const { node, root } = await panel({ spans: "0 100" });
    const video = findAll(root, (n) => n.tagName === "video")[0];
    video.paused = true;
    const bar = findAll(root, (n) => n.style.cssText.includes("height:44px"))[0];
    const region = findAll(bar, (n) =>
      n.style.cssText.includes("cursor:grab"))[0];
    let writes = 0;
    get(node, "spans").callback = () => { writes++; };
    region.fire("pointerdown", { clientX: 0 });
    global.window.fire("pointermove", { clientX: 0 });
    // 400px over 300 frames is 1.33px per frame, so 1px already crosses one.
    // 0.6px rounds back to the same frame, which is the case being tested.
    global.window.fire("pointermove", { clientX: 0.6 });
    check("no write for a sub-frame move", writes, 0);
    global.window.fire("pointermove", { clientX: 100, buttons: 1 });
    check("and one for a real one", writes, 1);
    global.window.fire("pointerup", {});
  }

  console.log("a drag whose release was missed does not live on forever");
  {
    // THE ZOOM REPORT. If a pointerup is ever missed -- released outside the
    // window, swallowed while the element was hidden, taken by another handler
    // -- the window listeners stay attached and `drag` stays set. Every later
    // pointermove then moves the trim, and a trackpad emits those while
    // zooming. `buttons === 0` is the self-heal: no button down, no drag.
    const { node, root } = await panel({ spans: "40 60 keep" });
    const bar = findAll(root, (n) => n.style.cssText.includes("height:44px"))[0];
    const region = findAll(bar, (n) =>
      n.style.cssText.includes("cursor:grab"))[0];

    region.fire("pointerdown", { clientX: 0, pointerId: 1 });
    global.window.fire("pointermove", { clientX: 250, buttons: 0, pointerId: 1 });
    check("a buttonless move changes nothing",
          get(node, "spans").value, "40 60 keep");
    global.window.fire("pointermove", { clientX: 300, buttons: 1, pointerId: 1 });
    check("and the stale drag is gone, not merely skipped",
          get(node, "spans").value, "40 60 keep");
  }

  console.log("a second pointer cannot fight the first over the same span");
  {
    const { node, root } = await panel({ spans: "0 100" });
    const bar = findAll(root, (n) => n.style.cssText.includes("height:44px"))[0];
    const region = findAll(bar, (n) =>
      n.style.cssText.includes("cursor:grab"))[0];
    region.fire("pointerdown", { clientX: 0, pointerId: 1 });
    global.window.fire("pointermove", { clientX: 100, buttons: 1, pointerId: 2 });
    check("the other pointer is ignored", get(node, "spans").value, "0 100");
    global.window.fire("pointermove", { clientX: 100, buttons: 1, pointerId: 1 });
    check("the one that started it still works",
          get(node, "spans").value, "75 100");
    global.window.fire("pointerup", {});
  }

  console.log("a cancelled gesture ends the drag too");
  {
    const { node, root } = await panel({ spans: "0 100" });
    const bar = findAll(root, (n) => n.style.cssText.includes("height:44px"))[0];
    const region = findAll(bar, (n) =>
      n.style.cssText.includes("cursor:grab"))[0];
    region.fire("pointerdown", { clientX: 0, pointerId: 1 });
    global.window.fire("pointercancel", {});
    global.window.fire("pointermove", { clientX: 250, buttons: 1, pointerId: 1 });
    check("nothing moved after the cancel", get(node, "spans").value, "0 100");
  }

  console.log("an unmeasurable timeline cannot move the trim");
  {
    // RESTORED: a range deletion of mine took these with it. A hidden or
    // collapsed element measures zero, and frameAt used to answer 0 -- a
    // perfectly good frame number, which every caller then acted on.
    const { node, root } = await panel({ spans: "40 60 keep" });
    const bar = findAll(root, (n) => n.style.cssText.includes("height:44px"))[0];
    const handles = findAll(bar, (n) =>
      n.style.cssText.includes("cursor:ew-resize"));
    const region = findAll(bar, (n) =>
      n.style.cssText.includes("cursor:grab"))[0];

    bar._rect = { left: 0, width: 0, top: 0, height: 0 };
    region.fire("pointerdown", { clientX: 0, pointerId: 1 });
    global.window.fire("pointermove", { clientX: 250, buttons: 1, pointerId: 1 });
    global.window.fire("pointerup", {});
    check("a drag on a hidden bar changes nothing",
          get(node, "spans").value, "40 60 keep");

    handles[0].fire("pointerdown", { clientX: 0, pointerId: 1 });
    global.window.fire("pointermove", { clientX: 250, buttons: 1, pointerId: 1 });
    global.window.fire("pointerup", {});
    check("nor does one on a handle", get(node, "spans").value, "40 60 keep");

    bar._rect = { left: 0, width: 400, top: 0, height: 44 };
    region.fire("pointerdown", { clientX: 0, pointerId: 1 });
    global.window.fire("pointermove", { clientX: 100, buttons: 1, pointerId: 1 });
    global.window.fire("pointerup", {});
    check("once visible it drags normally again",
          get(node, "spans").value, "115 60 keep");
  }

  console.log("the animation loop NEVER resizes the node");
  {
    // THE THRASH. draw() rewrites the counter every frame; a watcher that
    // measured the panel and resized on the result turned that into a feedback
    // loop -- the node resized every frame while playing, the trim box
    // flickered away and the handles could not be grabbed because the bar's
    // rect was never stable. Height changes on EVENTS now, never on a tick.
    const node = fakeNode();
    let relaid = 0;
    node.__vtRelayout = () => { relaid++; };
    const root = M.attach(node, false);
    await new Promise((r) => setTimeout(r, 0));
    relaid = 0;

    root.scrollHeight = 600;
    for (let i = 0; i < 20; i++) global.__frame?.();
    check("twenty frames, no resize", relaid, 0);

    root.scrollHeight = 900;
    for (let i = 0; i < 20; i++) global.__frame?.();
    check("not even when the panel really does change height", relaid, 0);
  }

  console.log("nothing resizes the node, ever");
  {
    // There is no relayout left to fire. The height is a constant and the
    // video's box is reserved, so loading a clip, learning its shape and
    // playing it all leave the node exactly as it was. Every bug in this area
    // came from something deciding the height had changed.
    const node = fakeNode();
    let relaid = 0;
    node.__vtRelayout = () => { relaid++; };
    const root = M.attach(node, false);
    await new Promise((r) => setTimeout(r, 0));
    const video = findAll(root, (n) => n.tagName === "video")[0];
    video.fire("loadedmetadata");
    for (let i = 0; i < 20; i++) global.__frame?.();
    check("not on load, not on metadata, not on a tick", relaid, 0);
  }

  console.log("the status row cannot reflow, whatever the counter says");
  {
    // if the row can wrap, a longer string adds a line and the panel gets
    // taller -- which is what let playback move the node at all
    const { root } = await panel({ spans: "0 100" });
    const status = findAll(root, (n) =>
      /flex-wrap:\s*nowrap/.test(n.style.cssText))[0];
    ok("it is nowrap", !!status);
    ok("with a fixed floor so an empty counter does not shrink it",
       /min-height:/.test(status.style.cssText));
    ok("and it clips rather than growing",
       /overflow:\s*hidden/.test(status.style.cssText));
  }

  console.log("the preview follows the widget, however the widget got its value");
  {
    // THE GRAPH-LOAD PATH. onNodeCreated runs BEFORE configure() restores the
    // saved widget values, so the panel loads whatever the combo defaults to --
    // the first file in the input folder -- and configure then sets the real
    // value SILENTLY, without firing the callback that would reload it. The
    // node then shows one clip while the widget names another, which is what
    // switching tabs and back produced.
    const node = fakeNode();
    let asked = [];
    global.fetch = async (url) => {
      const m = /file=([^&]+)/.exec(url);
      if (m) asked.push(decodeURIComponent(m[1]));
      return { json: async () => ({ ok: true, frames: 300, fps: 30,
                                    width: 640, height: 480, duration: 10 }) };
    };
    const root = M.attach(node, false);
    await new Promise((r) => setTimeout(r, 0));
    check("it loaded the widget's file", asked, ["clip.mp4"]);

    // configure() assigns the value directly -- no callback, exactly as the
    // frontend does when restoring a saved graph
    get(node, "file").value = "the_real_one.mp4";
    global.__frame?.();
    await new Promise((r) => setTimeout(r, 0));
    check("a silent change is noticed", asked,
          ["clip.mp4", "the_real_one.mp4"]);

    const video = findAll(root, (n) => n.tagName === "video")[0];
    ok("and the preview points at it",
       (video.src || "").includes("the_real_one.mp4"));

    // and it settles: no reloading on every frame afterwards
    for (let i = 0; i < 10; i++) global.__frame?.();
    await new Promise((r) => setTimeout(r, 0));
    check("then it stops asking", asked.length, 2);

    global.fetch = async () => ({
      json: async () => ({ ok: true, frames: 300, fps: 30, width: 640,
                           height: 480, duration: 10 }),
    });
  }

  console.log("the node can actually be CREATED");
  {
    // THE TEST THAT WAS MISSING. Everything here called attach() directly and
    // never went through the extension's onNodeCreated -- which is where the
    // widget's computeSize is defined, and where it referenced a variable that
    // does not exist in that scope. The suite was green while loading a
    // workflow threw ReferenceError before the node appeared at all.
    ok("the extension registered itself",
       typeof M.ext.beforeRegisterNodeDef === "function");

    const nodeType = function () {};
    nodeType.prototype = {};
    await M.ext.beforeRegisterNodeDef(nodeType, { name: "VideoTrimLoad" });
    ok("and hooked the node type",
       typeof nodeType.prototype.onNodeCreated === "function");

    const node = fakeNode();
    let threw = null;
    try {
      nodeType.prototype.onNodeCreated.call(node);
    } catch (e) {
      threw = e;
    }
    if (threw) console.log(`       ${threw.message}`);
    ok("creating one does not throw", threw === null);

    // and the widget's own sizing hook works when CALLED, which is what the
    // frontend does during configure()
    const w = node.__domWidget;
    ok("the panel widget was added", !!w);
    const size = w.computeSize(460);
    ok("its computeSize answers", Array.isArray(size) && size[1] > 0);
    check("with the width it was handed", size[0], 460);

    // TOO THIN TO LAY OUT. `setSize(this.computeSize())` passes computeSize's
    // own width through -- the node's natural minimum -- and at that width
    // every row wraps, the chrome is no longer the constant the height is
    // computed from, and the panel spills out of the node.
    check("it opens at least MIN_WIDTH wide", node.size[0] >= 460, true);

    // and cannot be dragged back into it
    const dragged = [200, node.size[1]];
    node.onResize(dragged);
    check("narrowing past the minimum is refused", dragged[0], 460);
    const wide = [900, node.size[1]];
    node.onResize(wide);
    check("but wider is fine, that is the point", wide[0], 900);
  }

  console.log("the height follows the WIDTH and the clip's aspect");
  {
    // CALLING THE REAL FUNCTION, not a copy of it beside the real one. The
    // first version of this test reimplemented the formula, so it passed while
    // the widget threw ReferenceError on a variable that was not in scope --
    // a test that agrees with itself and with nothing else.
    const CHROME = 285, MIN_VIDEO = 120, MAX_VIDEO = 520;
    const h = M.panelHeight;

    // 460 wide, 20 of gutter, 16:9: 440/1.778 = 248 of picture
    check("a 16:9 clip in a 460 node", h(460, 16 / 9), CHROME + 248);
    ok("wider node, taller node", h(900, 16 / 9) > h(460, 16 / 9));

    console.log("  and nothing about it depends on the node's own height");
    // this is the property that makes it safe: there is no height on the
    // right-hand side, so there is nothing to ratchet
    check("the same width always gives the same answer",
          h(460, 16 / 9), h(460, 16 / 9));

    console.log("  a portrait clip is capped, unlike VHS");
    // 440/0.5625 = 782 of picture, and with the controls that is a node you
    // cannot see what you are trimming in
    check("the picture stops at the cap", h(460, 9 / 16), CHROME + MAX_VIDEO);

    console.log("  and an unknown clip falls back rather than collapsing");
    check("no aspect yet, no guessing", h(460, 0), CHROME + MIN_VIDEO);
    // 160 of usable width at 16:9 is 90 of picture, which the floor lifts
    check("a very narrow node still shows a readable strip",
          h(60, 16 / 9), CHROME + MIN_VIDEO);
  }

  console.log("the panel FILLS its box, so the picture centres in the band");
  {
    // the panel still has to fill the box it is given: when the cap has taken
    // over, the band is taller than the picture and the picture centres in it
    const { root } = await panel({ spans: "0 100" });
    ok("the panel fills the height it is given",
       /height:100%/.test(root.style.cssText));
    ok("and counts its padding inside that",
       /box-sizing:border-box/.test(root.style.cssText));
    ok("it can shrink below its content", /min-height:0/.test(root.style.cssText));

    const video = findAll(root, (n) => n.tagName === "video")[0];
    const wrap = video.parentNode;
    ok("the preview flexes into what is left",
       /flex:1 1 auto/.test(wrap.style.cssText));
    ok("and centres the picture in it",
       /align-items:center/.test(wrap.style.cssText));
    ok("the video is bounded, not sized",
       /max-width:100%/.test(video.style.cssText) &&
       /max-height:100%/.test(video.style.cssText));

    // EVERY OTHER ROW IS FIXED. One left flexible gets squeezed by a tall
    // clip, and the row that vanished that way was the trim bar.
    const flexible = [...root.children].filter(
      (c) => c !== wrap && c.style.flex !== "0 0 auto");
    check("nothing else can be squeezed", flexible.length, 0);
    check("and the rows are all accounted for", root.children.length, 8);
  }

  console.log("learning a clip's shape resizes the node, once");
  {
    const node = fakeNode();
    let sized = 0;
    node.setSize = function (v) { sized++; this.size = v; };
    node.computeSize = () => [460, 533];
    const root = M.attach(node, false);
    await new Promise((r) => setTimeout(r, 0));
    const video = findAll(root, (n) => n.tagName === "video")[0];

    sized = 0;
    video.videoWidth = 1920;
    video.videoHeight = 1080;
    video.fire("loadedmetadata");
    check("the aspect is taken from the clip",
          Math.round(node.__vtAspect * 100), 178);
    check("and the node is resized for it", sized, 1);

    // the same clip again must not keep resizing: metadata can fire more than
    // once, and every one of those would fight a width the user had chosen
    video.fire("loadedmetadata");
    check("the same shape does not resize again", sized, 1);
  }

  console.log("the bar does not clip its own handles");
  {
    // THE HANDLES SIT AT left:0% AND left:100% with a negative margin, so they
    // straddle the edge. A clipping parent removes the outer half of each --
    // and at a full-clip span that leaves almost nothing to grab, which is why
    // the ends would not drag.
    const { root } = await panel({ spans: "0 100" });
    const bar = findAll(root, (n) => n.style.cssText.includes("height:44px"))[0];
    ok("the bar does not clip", !/overflow:\s*hidden/.test(bar.style.cssText));
    const handles = findAll(bar, (n) =>
      n.style.cssText.includes("cursor:ew-resize"));
    check("both handles are there", handles.length, 2);
    for (const h of handles) {
      ok("with a grab area wider than the mark",
         /width:15px/.test(h.style.cssText));
      // without this a gesture starting on a handle can be claimed as a scroll
      // and the drag never receives its moves
      ok("and touch-action none, so the gesture is ours",
         /touch-action:\s*none/.test(h.style.cssText));
    }
    ok("the bar claims its gestures too",
       /touch-action:\s*none/.test(bar.style.cssText));
  }

  console.log("the upload control reads as a stock widget row");
  {
    const { root } = await panel({ spans: "0 100" });
    const open = findAll(root, (n) => n.tagName === "label")[0];
    ok("it is the first thing in the panel", root.children[0] === open);
    check("labelled as the button it replaces", open.textContent,
          "choose file to upload");
    ok("full width, like a widget row",
       /width:\s*100%/.test(open.style.cssText));
    ok("and centred in it", /text-align:\s*center/.test(open.style.cssText));
    // it has to read as a BUTTON against the node body, not as a panel of the
    // same colour: darker, with an edge
    ok("darker than the node body", /background:#1e1e1e/.test(open.style.cssText));
    ok("and it has a border", /border:1px solid/.test(open.style.cssText));
  }

  console.log("the video keeps its own aspect, rather than being boxed and barred");
  {
    // A <video> with auto dimensions sizes itself to its own ratio; max-width
    // and max-height then shrink it while KEEPING that ratio, as an <img>
    // does. Forcing width:100% and letting object-fit letterbox inside the
    // leftover box is what put black bars down the sides of a portrait clip.
    const { root } = await panel({ spans: "0 100" });
    const video = findAll(root, (n) => n.tagName === "video")[0];
    const css = video.style.cssText;
    // anchored, or it matches the tail of `max-width:100%` and passes on the
    // very thing it is checking for
    ok("its width is not forced", !/(^|;)\s*width:\s*100%/.test(css));
    ok("and it is not letterboxed inside a box", !/object-fit/.test(css));
    ok("it is bounded rather than sized", /max-width:\s*100%/.test(css) &&
                                          /max-height:/.test(css));
    // centred by its WRAPPER now -- a flex box of reserved height -- rather
    // than by a margin on the video itself
    ok("and centred by the box that reserves its space",
       /justify-content:center/.test(video.parentNode.style.cssText));
  }

  console.log("editing the text widget by hand still wins");
  {
    const { node } = await panel({ spans: "0 100" });
    const w = get(node, "spans");
    w.value = "20 30 typed\n90 10";
    w.callback(w.value);
    check("the timeline followed the text", w.value, "20 30 typed\n90 10");
  }

  console.log();
  if (fails) { console.log(`FAIL — ${fails} check(s)`); process.exit(1); }
  console.log("timeline: ends resize, the middle slides, and nothing squashes");
})();
