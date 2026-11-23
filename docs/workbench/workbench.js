"use strict";

const demo = window.workbenchData;
const byId = (id) => document.getElementById(id);
const byName = (id, value) => { byId(id).textContent = value; };
const pageSize = 12;
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

function ratio(value) {
  const [numerator, denominator] = String(value).split("/").map(Number);
  return numerator / (denominator ?? 1);
}

function format(value, kind) {
  if (value == null || value === "") return "—";
  if (formatters[kind]) return formatters[kind].format(value);
  if (kind === "bps") return formatters.fixed2.format(value) + " bp";
  if (kind === "nsMs") return formatters.number.format(value / 1e6) + " ms";
  if (kind === "millionCny") return formatters.fixed2.format(value / 1e6) + " 百万元";
  if (kind === "rationalUsd") return formatters.usd.format(ratio(value) / 100);
  if (kind === "rationalPercent") return formatters.percent.format(ratio(value));
  return String(value);
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
  const formula = typeof value === "string" && /^[\s\u0000-\u001f]*[=+\-@]/u.test(raw.normalize("NFKC"));
  const safe = formula ? "'" + raw : raw;
  return '"' + safe.replaceAll('"', '""') + '"';
}

function renderGraph(container, graph) {
  const canvas = node("canvas", "graph");
  canvas.width = 960;
  canvas.height = 460;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", graph.summary);
  container.append(canvas, node("p", "graph-key", graph.key));
  const context = canvas.getContext("2d");
  if (!context) return;

  const points = new Map(graph.nodes.map((item) => [item.id, {
    x: 62 + item.x * 836,
    y: 53 + item.y * 354,
  }]));
  for (const edge of graph.edges) {
    const source = points.get(edge.from);
    const target = points.get(edge.to);
    const angle = Math.atan2(target.y - source.y, target.x - source.x);
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    const offset = edge.reverse ? 6 : 0;
    const x = target.x - dx * 29 - dy * offset;
    const y = target.y - dy * 29 + dx * offset;
    context.strokeStyle = edge.short ? "#af3c4b" : "#56788b";
    context.lineWidth = edge.short ? 4 : 2.5;
    context.beginPath();
    context.moveTo(source.x + dx * 29 - dy * offset, source.y + dy * 29 + dx * offset);
    context.lineTo(x, y);
    context.lineTo(x - dx * 13 - dy * 6, y - dy * 13 + dx * 6);
    context.moveTo(x, y);
    context.lineTo(x - dx * 13 + dy * 6, y - dy * 13 - dx * 6);
    context.stroke();
  }

  const colors = {
    direct: ["#a93047", "#ffffff"],
    cascade: ["#f8e1bb", "#663800"],
    solvent: ["#e0eee9", "#164f43"],
    outside: ["#dfe8f7", "#193f6b"],
  };
  for (const item of graph.nodes) {
    const { x, y } = points.get(item.id);
    const [fill, ink] = colors[item.status];
    context.beginPath();
    context.arc(x, y, 27, 0, 2 * Math.PI);
    context.fillStyle = fill;
    context.fill();
    context.strokeStyle = ink;
    context.lineWidth = 1.5;
    context.stroke();
    context.fillStyle = ink;
    context.font = "700 23px system-ui";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(item.label ?? item.id, x, y);
  }
}

function renderBars(container, bars, view, selected, choose) {
  if (bars.length === 0) {
    container.append(node("p", "empty", "当前选择没有可展示的数据。"));
    return;
  }

  const reference = view.reference ?? 0;
  const values = bars.map(({ value }) => value);
  const min = Math.min(view.min ?? 0, reference, ...values);
  const max = Math.max(view.max ?? 0, reference, ...values);
  const position = (value) => 100 * (value - min) / (max - min || 1);
  const list = node("div", "chart-bars");
  for (const item of bars) {
    const row = item.action ? button(null, "bar-row") : node("div", "bar-row");
    if (item.action) {
      row.setAttribute("aria-pressed", String(selected(item.action)));
      row.dataset.bar = item.action.key;
      row.addEventListener("click", () => choose(item.action));
    }
    const track = node("span", "bar-track");
    track.setAttribute("aria-hidden", "true");
    const segment = node("span", "bar-segment");
    segment.style.left = Math.min(position(reference), position(item.value)) + "%";
    segment.style.width = Math.abs(position(item.value) - position(reference)) + "%";
    track.append(segment);
    if (min < reference && reference < max) {
      const marker = node("span", "bar-marker");
      marker.style.left = position(reference) + "%";
      track.append(marker);
    }
    row.append(node("span", "bar-label", item.label), track,
      node("span", "bar-value", format(item.value, view.format)));
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
  const labels = (key, value) => app.filters.find((item) => item.key === key)?.labels?.[value] ?? value;
  const options = new Map(app.filters.map(({ key }) => [
    key, [...new Set(app.records.map((record) => String(record.values[key])))],
  ]));
  const filterState = new Map(app.filters.map(({ key }) => {
    const value = parameters.get(key) ?? "";
    return [key, options.get(key).includes(value) ? value : ""];
  }));
  const state = {
    query: (parameters.get("q") ?? "").slice(0, 140),
    filters: filterState,
    order: app.orders.some(({ id }) => id === parameters.get("sort")) ? parameters.get("sort") : "",
    view: app.views.findIndex(({ id }) => id === parameters.get("view")),
    record: parameters.get("record") ?? app.initialRecord,
    page: 0,
  };
  if (state.view < 0) state.view = 0;

  function matching() {
    const terms = state.query.trim().toLocaleLowerCase().split(/\s+/u).filter(Boolean);
    const filtered = app.records.filter((record) => {
      for (const [key, value] of state.filters) {
        if (value && String(record.values[key]) !== value) return false;
      }
      if (terms.length === 0) return true;
      const text = [record.id, record.title, record.subtitle, record.search, ...Object.values(record.values)]
        .join(" ").toLocaleLowerCase();
      return terms.every((term) => text.includes(term));
    });
    const order = app.orders.find(({ id }) => id === state.order);
    if (order) filtered.sort((left, right) => {
      const a = left.values[order.key];
      const b = right.values[order.key];
      if (a == null) return b == null ? 0 : 1;
      if (b == null) return -1;
      const direction = order.descending ? -1 : 1;
      return direction * (typeof a === "number" ? a - b : String(a).localeCompare(String(b)));
    });
    return filtered;
  }

  function syncUrl() {
    const url = new URL(location.href);
    url.search = "";
    for (const [key, value] of state.filters) if (value) url.searchParams.set(key, value);
    if (state.query) url.searchParams.set("q", state.query);
    if (state.order) url.searchParams.set("sort", state.order);
    if (state.view) url.searchParams.set("view", app.views[state.view].id);
    if (state.record && state.record !== app.initialRecord) url.searchParams.set("record", state.record);
    try {
      history.replaceState(null, "", url);
    } catch (error) {
      if (error.name !== "SecurityError") throw error;
    }
  }

  function chartData(view, records, record) {
    if (view.mode === "group") {
      const groups = new Map();
      for (const item of records) {
        const value = String(item.values[view.group]);
        groups.set(value, (groups.get(value) ?? 0) + 1);
      }
      return options.get(view.group).filter((value) => groups.has(value)).map((value) => ({
        label: labels(view.group, value),
        value: groups.get(value),
        action: { filter: view.group, value, key: view.group + ":" + value },
      }));
    }
    if (view.mode === "detail") return record?.charts?.[view.id] ?? [];
    const results = records.filter((item) => Number.isFinite(item.values[view.metric]));
    if (view.descending) results.sort((a, b) => b.values[view.metric] - a.values[view.metric]);
    const limit = view.limit ?? 8;
    const shown = results.slice(0, limit);
    if (record && results.includes(record) && !shown.includes(record)) shown[shown.length - 1] = record;
    return shown.map((item) => ({
      label: item.chartLabel ?? item.title,
      value: item.values[view.metric],
      action: { record: item.id, key: item.id },
    }));
  }

  function chartSelected(action) {
    return action.record ? state.record === action.record : state.filters.get(action.filter) === action.value;
  }

  function chooseChart(action) {
    if (action.record) {
      state.record = action.record;
      state.page = Math.floor(matching().findIndex(({ id }) => id === action.record) / pageSize);
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
    container.replaceChildren();
    if (view.mode === "graph" && record?.graph) renderGraph(container, record.graph);
    else renderBars(container, chartData(view, records, record), view, chartSelected, chooseChart);
    [...byId("views").children].forEach((item, index) => {
      item.setAttribute("aria-pressed", String(index === state.view));
    });
  }

  function chooseRecord(id) {
    state.record = id;
    refresh();
    if (matchMedia("(max-width: 940px)").matches) {
      byId("detail").focus();
    } else {
      [...byId("rows").querySelectorAll("button")].find((item) => item.dataset.id === id)
        ?.focus({ preventScroll: true });
    }
  }

  function renderDetail(record) {
    const container = byId("record-detail");
    container.replaceChildren();
    byId("download-record").disabled = !record;
    if (!record) {
      container.append(node("p", "empty", "筛选后可选择一条记录查看详情。"));
      return;
    }
    container.append(node("h3", "record-heading", record.title), node("p", "record-caption", record.subtitle));
    if (record.facts?.length) container.append(renderFacts(record.facts));
    for (const section of record.sections ?? []) container.append(renderSection(section));
    for (const section of app.related?.[record.related] ?? []) container.append(renderSection(section));
  }

  function refresh() {
    const records = matching();
    const record = records.find(({ id }) => id === state.record) ?? records[state.page * pageSize] ?? records[0];
    state.record = record?.id ?? "";
    const maxPage = Math.max(0, Math.ceil(records.length / pageSize) - 1);
    state.page = Math.max(0, Math.min(state.page, maxPage));
    const start = state.page * pageSize;
    const rows = byId("rows");
    rows.replaceChildren();
    for (const entry of records.slice(start, start + pageSize)) {
      const row = node("tr");
      row.dataset.selected = String(entry.id === state.record);
      const first = node("th", "records-cell");
      first.scope = "row";
      const link = button(entry.title, "record-button");
      link.dataset.id = entry.id;
      link.setAttribute("aria-pressed", row.dataset.selected);
      link.addEventListener("click", () => chooseRecord(entry.id));
      first.append(link);
      row.append(first);
      for (const column of app.columns) {
        const value = entry.values[column.key];
        row.append(node("td", "records-cell", format(labels(column.key, value), column.format)));
      }
      rows.append(row);
    }
    byId("empty").hidden = records.length > 0;
    byId("export").disabled = records.length === 0;
    byId("previous").disabled = state.page === 0;
    byId("next").disabled = state.page === maxPage;
    byName("count", records.length
      ? `第 ${start + 1}–${Math.min(start + pageSize, records.length)} 条 · 共 ${records.length} 条`
      : "共 0 条");
    renderChart(records, record);
    renderDetail(record);
    syncUrl();
  }

  function field(text, input, className = "field") {
    const label = node("label", className);
    label.append(node("span", null, text), input);
    return label;
  }

  document.title = app.project + " · " + app.title;
  for (const [name, color] of Object.entries(app.theme)) {
    document.documentElement.style.setProperty("--" + name, color);
  }
  const texts = {
    "brand-mark": app.mark, brand: app.project, "data-kind": app.dataKind,
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
  search.maxLength = 140;
  search.value = state.query;
  search.placeholder = app.searchPlaceholder;
  search.addEventListener("input", () => {
    state.query = search.value;
    state.page = 0;
    refresh();
  });
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
    state.record = app.initialRecord;
    state.page = 0;
    refresh();
  });

  function changePage(offset) {
    state.page += offset;
    state.record = matching()[state.page * pageSize]?.id;
    refresh();
  }
  byId("previous").addEventListener("click", () => changePage(-1));
  byId("next").addEventListener("click", () => changePage(1));
  byId("export").addEventListener("click", () => {
    const header = [app.recordLabel, ...app.columns.map(({ title, exportTitle }) => exportTitle ?? title)];
    const records = matching().map((record) => [
      record.title, ...app.columns.map(({ key }) => record.values[key]),
    ]);
    const csv = [header, ...records].map((row) => row.map(csvCell).join(",")).join("\r\n");
    download("\ufeff" + csv + "\r\n", app.slug + "-selection.csv", "text/csv;charset=utf-8");
  });
  byId("download-record").addEventListener("click", () => {
    const record = app.records.find(({ id }) => id === state.record);
    if (!record) return;
    const data = {
      project: app.project, sources: app.sources, record,
      related: app.related?.[record.related] ?? [],
    };
    const id = record.id.replace(/[^a-z0-9_-]+/gi, "-");
    download(JSON.stringify(data, null, 2) + "\n", app.slug + "-" + id + ".json", "application/json");
  });
  byId("workspace").hidden = false;
  const selectedIndex = matching().findIndex(({ id }) => id === parameters.get("record"));
  if (selectedIndex >= 0) state.page = Math.floor(selectedIndex / pageSize);
  refresh();
}

if (demo?.records?.length) init(demo);
else byId("startup-error").hidden = false;
