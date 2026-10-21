// panel.js
// Generates the control panel from SCHEMA (src/sim/params.js). Each entry is a
// range slider + numeric readout, grouped by section. Debug overlays are
// individual checkboxes so they can be toggled one at a time.

import { SCHEMA } from "../sim/params.js";

export const DEBUG_KEYS = ["heading", "vision", "hitbox", "bbox", "hashCells", "bloom"];
export const DEBUG_LABELS = {
  heading: "Heading vectors",
  vision: "Vision radius",
  hitbox: "Hitboxes",
  bbox: "Bounding boxes",
  hashCells: "Spatial-hash cells",
  bloom: "Bloom (glow)",
};

// Builds the panel DOM. `onChange(key, value)` fires for param sliders;
// `onDebug(key, value)` fires for the debug checkboxes; `onAction(name)` for
// buttons (reset / pause).
export function buildPanel(rootEl, params, { onChange, onDebug, onAction, debug }) {
  rootEl.innerHTML = "";
  const inputs = {}; // key -> { range, number }

  // ---- Actions ----
  const actions = el("div", "panel-actions");
  actions.append(
    button("Pause / Resume", () => onAction("pause")),
    button("Reset world", () => onAction("reset")),
    button("Randomize seed", () => onAction("reseed")),
  );
  rootEl.append(actions);

  // ---- Param groups ----
  const groups = new Map();
  for (const [group, key, label, min, max, step] of SCHEMA) {
    if (!groups.has(group)) {
      const sec = section(group);
      groups.set(group, sec);
      rootEl.append(sec.wrap);
    }
    const body = groups.get(group).body;
    const row = el("label", "panel-row");
    const nameSpan = el("span", "panel-row-label");
    nameSpan.textContent = label;
    const range = document.createElement("input");
    range.type = "range";
    range.min = String(min);
    range.max = String(max);
    range.step = String(step);
    range.value = String(params[key]);
    const num = document.createElement("input");
    num.type = "number";
    num.min = String(min);
    num.max = String(max);
    num.step = String(step);
    num.value = String(round3(params[key]));
    num.className = "panel-num";

    range.addEventListener("input", () => {
      const v = parseFloat(range.value);
      num.value = String(round3(v));
      onChange(key, v);
    });
    num.addEventListener("change", () => {
      let v = parseFloat(num.value);
      if (Number.isNaN(v)) v = params[key];
      v = Math.min(max, Math.max(min, v));
      num.value = String(round3(v));
      range.value = String(v);
      onChange(key, v);
    });

    inputs[key] = { range, number: num };
    row.append(nameSpan, range, num);
    body.append(row);
  }

  // ---- Debug overlays ----
  const dbgSec = section("Debug overlays");
  rootEl.append(dbgSec.wrap);
  const boxes = {};
  for (const key of DEBUG_KEYS) {
    const row = el("label", "panel-check");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = !!debug[key];
    box.addEventListener("change", () => onDebug(key, box.checked));
    const lbl = el("span", "panel-row-label");
    lbl.textContent = DEBUG_LABELS[key];
    row.append(box, lbl);
    dbgSec.body.append(row);
    boxes[key] = box;
  }

  function sync(p) {
    for (const [key, ctl] of Object.entries(inputs)) {
      if (document.activeElement === ctl.number) continue;
      ctl.range.value = String(p[key]);
      ctl.number.value = String(round3(p[key]));
    }
  }

  return { sync, inputs, boxes };
}

function el(tag, cls) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  return n;
}

function section(title) {
  const body = el("div", "panel-section");
  const h = el("h3", "panel-section-title");
  h.textContent = title;
  const wrap = el("div", "panel-section-body");
  wrap.append(h, body);
  return { body, wrap, head: h };
}

function button(label, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  b.className = "panel-btn";
  b.addEventListener("click", onClick);
  return b;
}

function round3(v) {
  return Math.round(v * 1000) / 1000;
}
