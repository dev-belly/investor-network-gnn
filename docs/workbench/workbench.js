"use strict";

const byId = (id) => document.getElementById(id);
const byName = (id, value) => {
  const element = byId(id);
  if (element.textContent !== value) element.textContent = value;
};
const pageSize = 12;
const maxQueryLength = 140;
const percentagePoints = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 2,
  signDisplay: "exceptZero",
});
const formatters = {
  integer: new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 }),
  number: new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 3 }),
  fixed2: new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  fixed3: new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 3, maximumFractionDigits: 3 }),
  fixed4: new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 4, maximumFractionDigits: 4 }),
  percent: new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 2 }),
  cny: new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY" }),
  usd: new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }),
};

function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text != null) result.textContent = text;
  return result;
}

function button(text, className) {
  const result = node("button", className, text);
  result.type = "button";
  return result;
}

function fraction(value) {
  const parts = /^(-?\d+)(?:\/([1-9]\d*))?$/u.exec(String(value));
  return parts ? [BigInt(parts[1]), BigInt(parts[2] ?? "1")] : null;
}

function compareValues(left, right, kind) {
  if (kind === "rationalUsd" || kind === "rationalPercent") {
    const a = fraction(left);
    const b = fraction(right);
    if (a && b) {
      const difference = a[0] * b[1] - b[0] * a[1];
      return difference < 0n ? -1 : Number(difference > 0n);
    }
  }
  return typeof left === "number" ? left - right : String(left).localeCompare(String(right));
}

function roundedFraction(value, scale) {
  const parsed = fraction(value);
  if (!parsed) return null;
  const [unscaled, denominator] = parsed;
  const numerator = unscaled * scale;
  const magnitude = numerator < 0n ? -numerator : numerator;
  let result = magnitude / denominator;
  const twiceRemainder = (magnitude % denominator) * 2n;
  // Match NetworkClear's exact, round-half-even display contract.
  if (twiceRemainder > denominator || (twiceRemainder === denominator && result % 2n === 1n)) {
    result += 1n;
  }
  return numerator < 0n ? -result : result;
}

function queryValue(value) {
  return value.replace(/[\r\n]+/gu, " ").slice(0, maxQueryLength).replace(/[\uD800-\uDBFF]$/u, "");
}

function format(value, kind) {
  if (value == null || value === "") return "—";
  if (Object.hasOwn(formatters, kind)) return formatters[kind].format(value);
  if (kind === "bps") return formatters.fixed2.format(value) + " bp";
  if (kind === "nsMs") return formatters.number.format(value / 1e6) + " ms";
  if (kind === "millionCny") return formatters.fixed2.format(value / 1e6) + " 百万元";
  if (kind === "rationalUsd") {
    const cents = roundedFraction(value, 1n);
    if (cents === null) return "—";
    const absolute = cents < 0n ? -cents : cents;
    const sign = cents < 0n ? "-$" : "$";
    return sign + formatters.integer.format(absolute / 100n) + "." + String(absolute % 100n).padStart(2, "0");
  }
  if (kind === "rationalPercent") {
    const basisPoints = roundedFraction(value, 10000n);
    return basisPoints === null ? "—" : formatters.percent.format(Number(basisPoints) / 10000);
  }
  return String(value);
}

function formatDifference(reference, current, kind) {
  if (reference == null || current == null || reference === "" || current === "") return "—";
  if (kind === "rationalUsd" || kind === "rationalPercent") {
    const base = fraction(reference);
    const next = fraction(current);
    if (!base || !next) return "—";
    const numerator = next[0] * base[1] - base[0] * next[1];
    const denominator = next[1] * base[1];
    const difference = roundedFraction(numerator + "/" + denominator, kind === "rationalUsd" ? 1n : 10000n);
    if (kind === "rationalPercent") {
      return percentagePoints.format(Number(difference) / 100) + " 个百分点";
    }
    return (difference > 0n ? "+" : "") + format(difference, kind);
  }
  if (Number.isFinite(reference) && Number.isFinite(current)) {
    const difference = current - reference;
    if (!Number.isFinite(difference)) return "—";
    if (kind === "percent") return percentagePoints.format(difference * 100) + " 个百分点";
    const unit = kind ?? "number";
    const magnitude = format(Math.abs(difference), unit);
    if (magnitude === format(0, unit)) return magnitude;
    return (difference > 0 ? "+" : "−") + magnitude;
  }
  return reference === current ? "相同" : "不同";
}

function download(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function csvCell(value) {
  const raw = value == null ? "" : String(value);
  const formula = typeof value === "string" && /^[\p{White_Space}\p{Cc}\p{Cf}]*[=+\-@]/u.test(raw.normalize("NFKC"));
  const safe = formula ? "'" + raw : raw;
  return '"' + safe.replaceAll('"', '""') + '"';
}

function csvHeading({ title, exportTitle, format: kind }) {
  if (exportTitle) return exportTitle;
  const units = { percent: "比例", bps: "bp", millionCny: "元", cny: "元", usd: "USD" };
  return Object.hasOwn(units, kind) ? title + "（" + units[kind] + "）" : title;
}

function downloadCsv(rows, filename) {
  const csv = rows.map((row) => row.map(csvCell).join(",")).join("\r\n");
  download("\ufeff" + csv + "\r\n", filename, "text/csv;charset=utf-8");
}

function drawEdges(canvas, edges, positions, institution, radius) {
  const { width, height } = canvas.getBoundingClientRect();
  if (width === 0 || height === 0) return;
  const density = Math.min(window.devicePixelRatio || 1, 3);
  const pixelsWide = Math.round(width * density);
  const pixelsHigh = Math.round(height * density);
  if (canvas.width !== pixelsWide) canvas.width = pixelsWide;
  if (canvas.height !== pixelsHigh) canvas.height = pixelsHigh;
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(pixelsWide / width, 0, 0, pixelsHigh / height, 0, 0);
  context.clearRect(0, 0, width, height);
  context.lineJoin = "round";
  context.lineCap = "round";
  for (const emphasized of [false, true]) {
    for (const edge of edges) {
      const incident = edge.from === institution || edge.to === institution;
      if (incident !== emphasized) continue;
      const from = positions.get(edge.from);
      const to = positions.get(edge.to);
      if (!from || !to) continue;
      const startX = from.x * width;
      const startY = from.y * height;
      const endX = to.x * width;
      const endY = to.y * height;
      const distance = Math.hypot(endX - startX, endY - startY);
      const stop = radius + 4;
      if (distance <= stop * 2) continue;
      const dx = (endX - startX) / distance;
      const dy = (endY - startY) / distance;
      const offset = edge.reverse ? 5 : 0;
      const x = endX - dx * stop - dy * offset;
      const y = endY - dy * stop + dx * offset;
      context.strokeStyle = edge.short ? "#f0a0a4" : "#dcf3ae";
      context.globalAlpha = emphasized ? 1 : .54;
      context.lineWidth = emphasized ? 2.15 : 1.45;
      context.setLineDash(edge.short ? [6, 4] : []);
      context.beginPath();
      context.moveTo(startX + dx * stop - dy * offset, startY + dy * stop + dx * offset);
      context.lineTo(x, y);
      context.stroke();
      context.setLineDash([]);
      context.beginPath();
      context.moveTo(x - dx * 7 - dy * 3, y - dy * 7 + dx * 3);
      context.lineTo(x, y);
      context.lineTo(x - dx * 7 + dy * 3, y - dy * 7 - dx * 3);
      context.stroke();
    }
  }
  context.globalAlpha = 1;
}

function renderGraph(container, graph, institution, choose) {
  const figure = node("figure");
  const stage = node("div", "graph-stage");
  const canvas = node("canvas", "graph-canvas");
  canvas.width = 960;
  canvas.height = 464;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", graph.summary);
  stage.append(canvas);
  const positions = new Map(graph.nodes.map((item) => [item.id, {
    x: (67 + item.x * 826) / 960,
    y: (52 + item.y * 360) / 464,
  }]));

  const readout = node("dl", "graph-readout");
  readout.setAttribute("aria-live", "polite");
  const status = { direct: "直接违约", cascade: "传染违约", solvent: "足额偿付" };
  const controls = new Map();
  let currentInstitution;
  function draw() {
    const radius = controls.get(currentInstitution)?.getBoundingClientRect().width / 2 || 0;
    drawEdges(canvas, graph.edges, positions, currentInstitution, radius);
  }
  function selectInstitution(item) {
    currentInstitution = item.id;
    for (const [id, control] of controls) {
      control.setAttribute("aria-pressed", String(id === item.id));
    }
    const facts = [
      ["机构状态", item.id + " · " + status[item.status]],
      ["实际付款", format(item.paid, "rationalUsd")],
      ["名义负债", format(item.nominal, "rationalUsd")],
      ["回收率", format(item.recovery, "rationalPercent")],
    ];
    readout.replaceChildren(...facts.map(([label, value]) => {
      const term = node("div");
      term.append(node("dt", null, label), node("dd", null, value));
      return term;
    }));
    draw();
  }
  for (const item of graph.nodes) {
    const { x, y } = positions.get(item.id);
    const control = button(item.id, "graph-node");
    control.dataset.status = item.status;
    control.style.left = 100 * x + "%";
    control.style.top = 100 * y + "%";
    control.setAttribute("aria-label", "机构 " + item.id + "，" + status[item.status]);
    control.addEventListener("click", () => {
      selectInstitution(item);
      choose(item.id);
    });
    controls.set(item.id, control);
    stage.append(control);
  }
  const selected = graph.nodes.find((item) => item.id === institution)
    ?? graph.nodes.find((item) => item.status === "direct")
    ?? graph.nodes[0];
  figure.append(stage, readout, node("figcaption", "graph-key", graph.key + " 所选机构的收付边会高亮。"));
  container.append(figure);
  selectInstitution(selected);
  const observer = new ResizeObserver(draw);
  observer.observe(stage);
  return () => observer.disconnect();
}

function renderBars(container, bars, view, domain, selected, choose) {
  if (bars.length === 0) {
    container.append(node("p", "empty", view.empty ?? "当前筛选没有可绘制的记录。"));
    return;
  }

  const { min, max, reference } = domain;
  const position = (value) => 100 * (value - min) / (max - min || 1);
  const list = node("div", "chart-bars");
  const scale = node("div", "chart-scale chart-columns");
  const axis = node("div", "axis-values");
  scale.setAttribute("aria-hidden", "true");
  axis.append(node("span", null, format(min, view.format)), node("span", null, format(max, view.format)));
  scale.append(node("span"), axis, node("span"));
  list.append(scale);
  for (const item of bars) {
    const row = item.action ? button(null, "bar-row chart-columns") : node("div", "bar-row chart-columns");
    if (item.action) {
      row.setAttribute("aria-pressed", String(selected(item.action)));
      row.dataset.bar = item.action.key;
      row.addEventListener("click", () => choose(item.action));
    }
    const track = node("span", "bar-track");
    track.setAttribute("aria-hidden", "true");
    const segment = node("span", "bar-segment");
    segment.dataset.direction = item.value < reference ? "negative" : "positive";
    segment.style.left = Math.min(position(reference), position(item.value)) + "%";
    segment.style.width = Math.abs(position(item.value) - position(reference)) + "%";
    track.append(segment);
    if (min < reference && reference < max) {
      const marker = node("span", "bar-marker");
      marker.style.left = position(reference) + "%";
      track.append(marker);
    }
    const label = item.displayValue === undefined
      ? format(item.value, view.format)
      : format(item.displayValue, view.displayFormat);
    row.append(node("span", "bar-label", item.label), track, node("span", "bar-value", label));
    list.append(row);
  }
  container.append(list);
  if (view.note) container.append(node("p", "chart-note", view.note));
}

function renderFacts(facts) {
  const list = node("dl", "fact-list");
  for (const { label, value, format: kind } of facts) {
    const row = node("div");
    row.append(node("dt", null, label), node("dd", null, format(value, kind)));
    list.append(row);
  }
  return list;
}

function renderSection(section) {
  const block = node(section.collapsed ? "details" : "section", "record-section");
  block.append(node(section.collapsed ? "summary" : "h4", null, section.title));
  if (section.text) block.append(node("p", null, section.text));
  if (section.items?.length) {
    const list = node("ul");
    for (const item of section.items) list.append(node("li", null, item));
    block.append(list);
  }
  if (section.table) {
    const wrap = node("div", "table-wrap");
    wrap.tabIndex = 0;
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", section.title);
    const table = node("table");
    const heading = table.createTHead().insertRow();
    for (const label of section.table.columns) {
      const cell = node("th", null, label);
      cell.scope = "col";
      heading.append(cell);
    }
    const body = table.createTBody();
    for (const values of section.table.rows) {
      const row = body.insertRow();
      values.forEach((value, index) => {
        row.append(node("td", null, format(value, section.table.formats?.[index])));
      });
    }
    wrap.append(table);
    block.append(wrap);
  }
  return block;
}

function init(app) {
  const parameters = new URLSearchParams(location.search);
  const recordsById = new Map(app.records.map((record) => [record.id, record]));
  const askedRecord = parameters.get("record");
  const askedReference = parameters.get("compare");
  const filtersByKey = new Map(app.filters.map((filter) => [filter.key, filter]));
  const labels = (key, value) => filtersByKey.get(key)?.labels?.[value] ?? value;
  const options = new Map(app.filters.map(({ key }) => [
    key, [...new Set(app.records.map((record) => String(record.values[key])))],
  ]));
  const searchIndex = new Map(app.records.map((record) => [
    record.id,
    [record.id, record.title, record.subtitle, record.search,
      ...Object.entries(record.values).flatMap(([key, value]) => [value, labels(key, value)])]
      .join(" ").normalize("NFKC").toLocaleLowerCase(),
  ]));
  const ownedKeys = ["q", "sort", "view", "record", "compare", ...options.keys()];
  const hasGraph = app.views.some(({ mode }) => mode === "graph");
  if (hasGraph) ownedKeys.push("node");
  const filterState = new Map(app.filters.map(({ key }) => {
    const value = parameters.get(key) ?? "";
    return [key, options.get(key).includes(value) ? value : ""];
  }));
  const state = {
    query: queryValue(parameters.get("q") ?? ""),
    filters: filterState,
    order: app.orders.some(({ id }) => id === parameters.get("sort")) ? parameters.get("sort") : "",
    view: app.views.findIndex(({ id }) => id === parameters.get("view")),
    record: askedRecord ?? app.initialRecord,
    reference: recordsById.get(askedReference) ?? null,
    institution: hasGraph ? parameters.get("node") ?? "" : "",
    page: 0,
  };
  if (state.view < 0) state.view = 0;
  let visibleRecords = [];
  let lastDetail = null;
  let disposeGraph = null;
  let renderedSearch = location.search;

  function matching(skippedFilter) {
    const terms = state.query.normalize("NFKC").trim().toLocaleLowerCase().split(/\s+/u).filter(Boolean);
    const filtered = app.records.filter((record) => {
      for (const [key, value] of state.filters) {
        if (key !== skippedFilter && value && String(record.values[key]) !== value) return false;
      }
      return terms.every((term) => searchIndex.get(record.id).includes(term));
    });
    if (skippedFilter) return filtered;
    const order = app.orders.find(({ id }) => id === state.order);
    if (order) filtered.sort((left, right) => {
      const a = left.values[order.key];
      const b = right.values[order.key];
      if (a == null) return b == null ? 0 : 1;
      if (b == null) return -1;
      const direction = order.descending ? -1 : 1;
      return direction * compareValues(a, b, order.format);
    });
    return filtered;
  }

  const linkNotices = [];
  if (askedRecord !== null && !recordsById.has(askedRecord)) {
    state.reference = null;
    linkNotices.push("链接指定的记录已不可用，请重新选择记录后再对照。");
  } else if (askedRecord !== null && !matching().some(({ id }) => id === askedRecord)) {
    state.query = "";
    for (const key of state.filters.keys()) state.filters.set(key, "");
    linkNotices.push("已清除与链接冲突的筛选条件，保留链接指定的记录。");
  }
  if (askedReference !== null && !recordsById.has(askedReference)) {
    linkNotices.push("链接指定的对照记录已不可用。");
  }

  function analysisUrl() {
    const url = new URL(location.href);
    for (const key of ownedKeys) url.searchParams.delete(key);
    for (const [key, value] of state.filters) if (value) url.searchParams.set(key, value);
    if (state.query) url.searchParams.set("q", state.query);
    if (state.order) url.searchParams.set("sort", state.order);
    if (state.view) url.searchParams.set("view", app.views[state.view].id);
    if (state.record && (state.reference || state.record !== app.initialRecord)) {
      url.searchParams.set("record", state.record);
    }
    if (state.reference) url.searchParams.set("compare", state.reference.id);
    if (state.institution) url.searchParams.set("node", state.institution);
    if (!state.reference && url.hash === "#comparison") url.hash = "detail";
    return url;
  }

  function syncUrl() {
    const url = analysisUrl();
    if (url.href === location.href) return;
    try {
      history.replaceState(history.state, "", url);
      renderedSearch = location.search;
    } catch (error) {
      if (error.name !== "SecurityError") throw error;
    }
  }

  window.addEventListener("popstate", () => {
    // Reopen the saved analysis when history changes the query, even within this document.
    if (location.search !== renderedSearch) location.reload();
  });
  function groupCounts(records, key) {
    const counts = new Map(options.get(key).map((value) => [value, 0]));
    for (const { values } of records) {
      const group = String(values[key]);
      counts.set(group, counts.get(group) + 1);
    }
    return counts;
  }

  const domains = new Map(app.views.filter(({ mode }) => mode !== "graph").map((view) => {
    let values;
    if (view.mode === "group") values = [...groupCounts(app.records, view.group).values()];
    else if (view.mode === "detail") values = app.records.flatMap(
      (record) => (record.charts?.[view.id] ?? []).map((bar) => bar.value),
    );
    else values = app.records.map((record) => record.values[view.metric]);
    const finite = values.filter(Number.isFinite);
    const reference = view.reference ?? 0;
    return [view.id, {
      reference,
      min: Math.min(view.min ?? 0, reference, ...finite),
      max: Math.max(view.max ?? 0, reference, ...finite),
    }];
  }));

  function chartData(view, records, record) {
    if (view.mode === "group") {
      const counts = groupCounts(matching(view.group), view.group);
      return [...counts].map(([value, count]) => ({
        label: labels(view.group, value),
        value: count,
        action: { filter: view.group, value, key: view.group + ":" + value },
      }));
    }
    if (view.mode === "detail") {
      return (record?.charts?.[view.id] ?? []).filter((bar) => Number.isFinite(bar.value));
    }
    const results = records.filter((item) => Number.isFinite(item.values[view.metric]));
    const sortKey = view.displayKey ?? view.metric;
    if (view.descending) results.sort((a, b) => compareValues(b.values[sortKey], a.values[sortKey], view.displayFormat));
    const limit = view.limit ?? 8;
    const shown = results.slice(0, limit);
    if (record && results.includes(record) && !shown.includes(record)) shown[shown.length - 1] = record;
    return shown.map((item) => ({
      label: item.chartLabel ?? item.title,
      value: item.values[view.metric],
      displayValue: item.values[view.displayKey],
      action: { record: item.id, key: item.id },
    }));
  }

  function chartSelected(action) {
    return action.record ? state.record === action.record : state.filters.get(action.filter) === action.value;
  }

  function selectRecord(id) {
    if (state.record !== id) state.institution = "";
    state.record = id;
  }

  function chooseChart(action) {
    if (action.record) {
      selectRecord(action.record);
    } else {
      const value = chartSelected(action) ? "" : action.value;
      state.filters.set(action.filter, value);
      filterFields.get(action.filter).value = value;
      state.page = 0;
    }
    refresh();
    [...byId("chart").querySelectorAll("[data-bar]")].find((item) => item.dataset.bar === action.key)
      ?.focus({ preventScroll: true });
  }

  function renderChart(records, record) {
    const view = app.views[state.view];
    byName("chart-heading", view.title);
    byName("chart-caption", view.caption);
    const container = byId("chart");
    disposeGraph?.();
    disposeGraph = null;
    container.replaceChildren();
    if (view.mode === "graph") {
      if (record?.graph?.nodes.length) {
        disposeGraph = renderGraph(container, record.graph, state.institution, (id) => {
          state.institution = id;
          syncUrl();
        });
      } else container.append(node("p", "empty", "选择清算情景以查看网络。"));
    } else {
      renderBars(container, chartData(view, records, record), view, domains.get(view.id), chartSelected, chooseChart);
    }
    [...byId("views").children].forEach((item, index) => {
      item.setAttribute("aria-pressed", String(index === state.view));
    });
  }

  function selectedButton() {
    return [...byId("rows").querySelectorAll("button")].find((item) => item.dataset.id === state.record);
  }

  function chooseRecord(id) {
    selectRecord(id);
    refresh();
    if (matchMedia("(max-width: 970px)").matches) {
      byId("detail").focus();
    } else {
      selectedButton()?.focus({ preventScroll: true });
    }
  }

  function renderDetail(record) {
    if (record === lastDetail) return;
    lastDetail = record;
    byId("detail").scrollTop = 0;
    const container = byId("record-detail");
    container.replaceChildren();
    byId("download-record").disabled = !record;
    byId("return-records").disabled = !record;
    if (!record) {
      container.append(node("p", "empty", "筛选后可选择一条记录查看详情。"));
      return;
    }
    container.append(node("h3", "record-heading", record.title), node("p", "record-caption", record.subtitle));
    if (record.facts?.length) container.append(renderFacts(record.facts));
    for (const section of record.sections ?? []) container.append(renderSection(section));
    for (const section of app.related?.[record.related] ?? []) container.append(renderSection(section));
  }

  function comparisonRows(reference, record) {
    return app.columns.map(({ key, title, format: kind }) => {
      const baseline = reference.values[key];
      const current = record.values[key];
      return [title, format(labels(key, baseline), kind), format(labels(key, current), kind),
        formatDifference(baseline, current, kind)];
    });
  }

  function renderComparison(record) {
    const reference = state.reference;
    const panel = byId("comparison");
    const content = byId("comparison-content");
    const action = byId("pin-record");
    const ready = Boolean(record && reference && record !== reference);
    action.disabled = !record || record === reference;
    byName("pin-record", record && record === reference ? "已设为对照" : "设为对照");
    panel.hidden = !reference;
    byId("comparison-actions").hidden = !ready;
    byName("share-status", "");
    if (!reference) {
      content.replaceChildren();
      return;
    }
    const pair = node("dl", "comparison-pair");
    const items = ready ? [["对照", reference], ["当前", record]] : [["对照", reference]];
    for (const [label, entry] of items) {
      const card = node("div", "comparison-identity");
      card.append(node("dt", null, label), node("dd", null, entry.title));
      pair.append(card);
    }
    if (!ready) {
      content.replaceChildren(pair, node("p", "comparison-hint", "选择另一条记录查看差异。"));
      return;
    }
    const table = node("table");
    table.setAttribute("aria-label", "对照记录与当前记录的主要指标");
    const head = table.createTHead().insertRow();
    for (const title of ["指标", "对照", "当前", "变化"]) {
      const cell = node("th", null, title);
      cell.scope = "col";
      head.append(cell);
    }
    const body = table.createTBody();
    for (const [title, baseline, current, difference] of comparisonRows(reference, record)) {
      const row = body.insertRow();
      const label = node("th", null, title);
      label.scope = "row";
      row.append(label, node("td", null, baseline), node("td", null, current),
        node("td", "comparison-change", difference));
    }
    content.replaceChildren(pair, table, node("p", "comparison-hint",
      "变化以原值计算，等于当前减对照。百分比差值以百分点显示。"));
  }

  function refresh() {
    visibleRecords = matching();
    const records = visibleRecords;
    const maxPage = Math.max(0, Math.ceil(records.length / pageSize) - 1);
    const index = records.findIndex(({ id }) => id === state.record);
    if (index >= 0) state.page = Math.floor(index / pageSize);
    state.page = Math.max(0, Math.min(state.page, maxPage));
    const start = state.page * pageSize;
    const record = index >= 0 ? records[index] : records[start];
    selectRecord(record?.id ?? "");
    if (state.institution && !record?.graph?.nodes.some(({ id }) => id === state.institution)) {
      state.institution = "";
    }
    const rows = byId("rows");
    rows.replaceChildren();
    for (const entry of records.slice(start, start + pageSize)) {
      const row = node("tr");
      row.setAttribute("role", "row");
      row.dataset.selected = String(entry.id === state.record);
      const first = node("th", "records-cell");
      first.scope = "row";
      first.setAttribute("role", "rowheader");
      const link = button(entry.title, "record-button");
      link.dataset.id = entry.id;
      link.setAttribute("aria-pressed", row.dataset.selected);
      link.addEventListener("click", () => chooseRecord(entry.id));
      first.append(link);
      row.append(first);
      for (const column of app.columns) {
        const value = entry.values[column.key];
        const cell = node("td", "records-cell", format(labels(column.key, value), column.format));
        cell.setAttribute("role", "cell");
        cell.dataset.label = column.title;
        row.append(cell);
      }
      rows.append(row);
    }
    byId("empty").hidden = records.length > 0;
    byId("records-table").hidden = records.length === 0;
    byId("export").disabled = records.length === 0;
    byId("previous").disabled = state.page === 0;
    byId("next").disabled = state.page === maxPage;
    byName("count", records.length
      ? `第 ${start + 1}–${Math.min(start + pageSize, records.length)} 条 · 共 ${records.length} 条`
      : "共 0 条");
    renderChart(records, record);
    renderDetail(record);
    renderComparison(record);
    syncUrl();
  }

  function field(text, input, className = "field") {
    const label = node("label", className);
    label.append(node("span", null, text), input);
    return label;
  }

  document.title = app.project + " · " + app.title;
  document.querySelector('meta[name="description"]').content = app.description;
  for (const [name, color] of Object.entries(app.theme)) {
    document.documentElement.style.setProperty("--" + name, color);
  }
  const texts = {
    "brand-mark": app.mark, brand: app.project,
    "toolbar-label": app.workspace, "hero-kicker": app.dataKind, title: app.title,
    description: app.description, "hero-stat-label": app.hero.label, "hero-stat-value": app.hero.value,
    "hero-stat-note": app.hero.note, "records-heading": app.recordHeading,
    "detail-heading": app.detailHeading, "source-note": app.sourceNote,
  };
  for (const [id, value] of Object.entries(texts)) byName(id, value);
  byId("repository").href = app.repository;
  byId("metrics").append(...app.metrics.map(({ label, value, note }) => {
    const card = node("div", "metric");
    const description = node("dd", null, value);
    if (note) description.append(node("small", null, note));
    card.append(node("dt", null, label), description);
    return card;
  }));
  for (const { label, url } of app.sources) {
    const item = node("li");
    const link = node("a", null, label);
    link.href = url;
    item.append(link);
    byId("source-list").append(item);
  }

  const search = node("input");
  search.type = "search";
  search.maxLength = maxQueryLength;
  search.value = state.query;
  search.placeholder = app.searchPlaceholder;
  function searchUpdated() {
    const value = queryValue(search.value);
    if (search.value !== value) search.value = value;
    if (state.query === value) return;
    state.query = value;
    state.page = 0;
    refresh();
  }
  search.addEventListener("input", (event) => {
    if (!event.isComposing) searchUpdated();
  });
  search.addEventListener("compositionend", searchUpdated);
  byId("filters").append(field("搜索", search, "field search"));
  const filterFields = new Map();
  for (const definition of app.filters) {
    const select = node("select");
    select.append(new Option("全部", ""));
    for (const value of options.get(definition.key)) select.append(new Option(labels(definition.key, value), value));
    select.value = state.filters.get(definition.key);
    select.addEventListener("change", () => {
      state.filters.set(definition.key, select.value);
      state.page = 0;
      refresh();
    });
    filterFields.set(definition.key, select);
    byId("filters").append(field(definition.title, select));
  }
  const order = node("select");
  order.append(new Option("样例顺序", ""));
  for (const option of app.orders) order.append(new Option(option.title, option.id));
  order.value = state.order;
  order.addEventListener("change", () => {
    state.order = order.value;
    state.page = 0;
    refresh();
  });
  byId("filters").append(field("排序", order));
  for (const label of [app.recordLabel, ...app.columns.map(({ title }) => title)]) {
    const cell = node("th", null, label);
    cell.scope = "col";
    cell.setAttribute("role", "columnheader");
    byId("columns").append(cell);
  }
  app.views.forEach(({ label }, index) => {
    const option = button(label);
    option.addEventListener("click", () => {
      state.view = index;
      refresh();
    });
    byId("views").append(option);
  });
  byId("views").hidden = app.views.length < 2;
  byId("reset").addEventListener("click", () => {
    for (const [key, select] of filterFields) {
      state.filters.set(key, "");
      select.value = "";
    }
    state.query = search.value = "";
    state.order = order.value = "";
    selectRecord(app.initialRecord);
    state.institution = "";
    state.page = 0;
    refresh();
  });
  byId("pin-record").addEventListener("click", () => {
    const record = recordsById.get(state.record);
    if (!record) return;
    state.reference = record;
    renderComparison(record);
    byId("clear-comparison").focus({ preventScroll: true });
    syncUrl();
  });
  byId("clear-comparison").addEventListener("click", () => {
    state.reference = null;
    renderComparison(recordsById.get(state.record));
    (byId("pin-record").disabled ? search : byId("pin-record")).focus({ preventScroll: true });
    syncUrl();
  });

  const shareDialog = byId("share-dialog");
  const shareField = byId("share-url");
  const shareButton = byId("share-comparison");
  shareButton.hidden = !/^https?:$/u.test(location.protocol);
  shareButton.addEventListener("click", async () => {
    const reference = state.reference;
    const record = state.record;
    const stillSelected = () => state.record === record && state.reference === reference;
    const url = analysisUrl();
    url.hash = "comparison";
    try {
      await navigator.clipboard.writeText(url.href);
      if (stillSelected()) byName("share-status", "对照链接已复制");
    } catch {
      if (!stillSelected()) return;
      shareField.value = url.href;
      if (!shareDialog.open) shareDialog.showModal();
      shareField.focus();
      shareField.select();
    }
  });
  byId("export-comparison").addEventListener("click", () => {
    const reference = state.reference;
    const record = recordsById.get(state.record);
    if (!reference || !record || record === reference) return;
    const header = ["指标", "对照 · " + reference.title, "当前 · " + record.title, "变化"];
    downloadCsv([header, ...comparisonRows(reference, record)], app.slug + "-comparison.csv");
  });
  function evidence(record) {
    return { record, related: app.related?.[record.related] ?? [] };
  }

  byId("export-evidence").addEventListener("click", () => {
    const reference = state.reference;
    const record = recordsById.get(state.record);
    if (!reference || !record || record === reference) return;
    const bundle = {
      project: app.project, fields: app.columns, sources: app.sources,
      reference: evidence(reference), current: evidence(record),
    };
    download(JSON.stringify(bundle, null, 2) + "\n", app.slug + "-comparison.json", "application/json");
  });

  function changePage(offset) {
    const active = document.activeElement;
    state.page += offset;
    selectRecord("");
    refresh();
    if (active === byId("previous") && byId("previous").disabled) byId("next").focus();
    if (active === byId("next") && byId("next").disabled) byId("previous").focus();
  }
  byId("previous").addEventListener("click", () => changePage(-1));
  byId("next").addEventListener("click", () => changePage(1));
  byId("return-records").addEventListener("click", () => selectedButton()?.focus());
  byId("export").addEventListener("click", () => {
    const header = [app.recordLabel, ...app.columns.map(csvHeading)];
    const records = visibleRecords.map((record) => [
      record.title, ...app.columns.map(({ key }) => record.values[key]),
    ]);
    downloadCsv([header, ...records], app.slug + "-selection.csv");
  });
  byId("download-record").addEventListener("click", () => {
    const record = recordsById.get(state.record);
    if (!record) return;
    const data = {
      project: app.project, sources: app.sources, ...evidence(record),
    };
    const id = record.id.replace(/[^a-z0-9_-]+/gi, "-");
    download(JSON.stringify(data, null, 2) + "\n", app.slug + "-" + id + ".json", "application/json");
  });
  refresh();
  byName("link-notice", linkNotices.join(" "));
  byId("link-notice").hidden = linkNotices.length === 0;
  for (const id of ["hero-stat", "metrics", "workspace", "sources"]) byId(id).hidden = false;
  if (location.hash === "#comparison" && state.reference) {
    requestAnimationFrame(() => byId("comparison").scrollIntoView());
  }
}

try {
  const app = window.workbenchData;
  if (!app?.records?.length || !app.views?.length) throw new Error("No usable workbench snapshot");
  init(app);
} catch (error) {
  console.error("Workbench startup failed", error);
  byId("startup-error").hidden = false;
}
