// The UI's fill() helper, against the bug it shipped with.
//
// THE BUG, as CJ saw it: "drop down for clip length doesn't update when you
// change it. at least not visually on the drop down. it does change settings."
//
// A DOM option's `value` is ALWAYS a string. The frame-count lists carry `v` as
// a NUMBER, and the chosen value arrived as a string, so the old guard
//
//     items.some(i => i.v === chosen)        // 192 === "192"  ->  false
//
// never matched, the selection was never applied, and the <select> fell back to
// displaying its first option while the value behind it was perfectly correct.
// Three dropdowns had it: the scene inspector's length, and the plan section's
// chunk frames and carry.
//
// fill() is read out of ui/index.html rather than copied here, so this cannot
// pass against a stale copy of the function.
const fs = require("fs");

const html = fs.readFileSync(__dirname + "/ui/index.html", "utf8");
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const body = src.match(/function fill\(sel, items, chosen\) \{[\s\S]*?\n\}/);
if (!body) {
  console.error("FAIL could not find fill() in ui/index.html");
  process.exit(1);
}

// THE STUB HAS TO MIRROR THE REAL THING, which is the whole point here: a real
// DOM element COERCES `value` to a string on assignment, and an <option> built
// from a number comes back as "192". A stub that stores the raw number hides the
// very mismatch this test exists to catch -- the first version of this file did
// exactly that and reported failures that were its own.
function makeOption() {
  let v = "", t = "";
  return {
    set value(x) { v = String(x); },
    get value() { return v; },
    set textContent(x) { t = String(x); },
    get textContent() { return t; },
  };
}
function makeSelect() {
  const state = { options: [], _value: "" };
  return {
    options: state.options,
    set innerHTML(x) {
      if (x === "") { state.options.length = 0; state._value = ""; }
    },
    get innerHTML() { return ""; },
    appendChild(o) {
      state.options.push(o);
      if (state.options.length === 1) state._value = o.value;  // first = default
    },
    // a real select refuses a value no option carries
    set value(x) {
      const want = String(x);
      if (state.options.some(o => o.value === want)) state._value = want;
    },
    get value() { return state._value; },
  };
}
const document = { createElement: makeOption };

const fill = new Function("document", `${body[0]}; return fill;`)(document);

let fails = 0;
const check = (name, got, want) => {
  const ok = got === want;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name}: ${JSON.stringify(got)}`);
  if (!ok) { fails++; console.log(`       wanted ${JSON.stringify(want)}`); }
};

const FRAMES = [39, 90, 141, 192, 243].map(r => ({ v: r, t: `${r}f` }));

console.log("a NUMBER option selected by a STRING — the bug");
let sel = makeSelect();
fill(sel, FRAMES, "192");
check("string chosen matches a numeric option", sel.value, "192");

console.log("and the other way round, since both forms are in use");
sel = makeSelect();
fill(sel, FRAMES, 192);
check("number chosen matches too", sel.value, "192");

console.log("plain string lists keep working");
sel = makeSelect();
fill(sel, ["euler", "res_multistep", "lcm"], "lcm");
check("string list, string chosen", sel.value, "lcm");

console.log("a value that is not in the list leaves the first option showing");
sel = makeSelect();
fill(sel, FRAMES, 777);
check("absent chosen does not blank the select", sel.value, "39");

console.log("and an empty choice is not treated as a value");
sel = makeSelect();
fill(sel, FRAMES, "");
check("empty chosen leaves the default", sel.value, "39");
sel = makeSelect();
fill(sel, FRAMES, null);
check("null chosen leaves the default", sel.value, "39");

console.log("refilling replaces the options rather than appending");
sel = makeSelect();
fill(sel, FRAMES, 90);
fill(sel, FRAMES, 141);
check("option count after two fills", sel.options.length, FRAMES.length);
check("and the second choice won", sel.value, "141");

console.log();
if (fails) { console.log(`FAIL — ${fails} check(s)`); process.exit(1); }
console.log("ui select: a number option can be chosen by a string");
