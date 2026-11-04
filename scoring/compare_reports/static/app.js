const $ = (id) => document.getElementById(id);
const labels = ["A", "B", "C", "D", "E"];
const coreSections = new Set([
  "Diagnosis",
  "Key Microscopic Findings",
  "Stage / Grade / Prognostic Features",
]);
const sectionOrder = [
  "Specimen / Site", "Diagnosis", "Tumor / Lesion Description",
  "Key Microscopic Findings", "Invasion / Necrosis / Margins",
  "IHC / Ancillary Studies", "Stage / Grade / Prognostic Features",
  "Differential Diagnoses", "Limitations", "Final Summary", "Report",
];
const state = {
  reader: null, cases: [], activeCase: null, meta: null, viewer: null,
  filter: "all", rankOrders: {}, rankValues: {}, viewerPinned: false, viewerCollapsed: false,
  viewerWindow: null,
  activeSlideId: null, caseRequestSeq: 0,
  dirty: false, autosaveTimer: null, autosaveRetryTimer: null, autosaveInFlight: false,
};
const viewerWindowChannelName = "pathxai-wsi-viewer-v1";
const viewerWindowStorageKey = "pathxai-wsi-viewer-case-v1";
const viewerWindowSelectionKey = "pathxai-wsi-viewer-selection-v1";
const viewerWindowChannel = "BroadcastChannel" in window
  ? new BroadcastChannel(viewerWindowChannelName)
  : null;
const readerKey = "pathxai_tutorial_compare_session_v1";
const draftStoragePrefix = "pathxai_tutorial_compare_draft_v1";
// Keep the storage namespace so pre-update unsynced drafts can be recovered.
const rankScale = {min: 1, max: 5, direction: "lower_is_better", ties_allowed: true};
const autosaveDelayMs = 1200;
const autosaveRetryMs = 10000;

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs || 20000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(body.error || `HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Request timed out; the local draft is safe.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
function withReader(path) {
  const join = path.includes("?") ? "&" : "?";
  return `${path}${join}reader_id=${encodeURIComponent(state.reader.reader_id)}`;
}
function profileFromForm() {
  const reader_id = $("readerId").value.trim() || crypto.randomUUID();
  $("readerId").value = reader_id;
  if (!/^[A-Za-z0-9_.-]{1,64}$/.test(reader_id)) throw new Error("Reader ID may contain letters, numbers, dots, underscores, and hyphens.");
  return {reader_id};
}
async function startReader() {
  try {
    if (state.dirty) writeLocalDraft();
    state.reader = profileFromForm();
    state.activeCase = null;
    state.activeSlideId = null;
    state.caseRequestSeq += 1;
    state.meta = null;
    state.rankOrders = {};
    state.rankValues = {};
    state.dirty = false;
    state.viewer?.close();
    $("slideSelect").replaceChildren(new Option("No case selected", ""));
    $("slideSelect").disabled = true;
    $("slideReviewNotice").hidden = true;
    $("slideReviewConfirmation").hidden = true;
    $("slideLoadStatus").textContent = "";
    $("sectionComparison").replaceChildren();
    $("reportGrid").replaceChildren();
    $("overallRankControl").replaceChildren();
    $("caseTitle").textContent = "Select a case";
    $("caseMeta").textContent = "Method identities are blinded.";
    localStorage.setItem(readerKey, JSON.stringify(state.reader));
    $("readerStatus").textContent = `Active: ${state.reader.reader_id}`;
    await loadCases();
  } catch (error) { alert(error.message); }
}
async function loadCases() {
  const status = await api("/api/status");
  state.cases = await api(withReader("/api/cases"));
  $("statusLine").textContent = `${status.case_count} cases · active local session ${state.reader.reader_id}`;
  renderCases();
  if (pendingLocalDrafts().length) scheduleAutosave(100);
}
function renderCases() {
  const query = $("caseSearch").value.trim().toLowerCase();
  const rows = state.cases.filter((row) => {
    const matches = state.filter === "all" || (state.filter === "slide_review" ? row.slide_review_required
      : row.status === state.filter);
    return matches && `${row.case_label} ${row.site}`.toLowerCase().includes(query);
  });
  $("caseList").replaceChildren(...rows.map((row) => {
    const button = document.createElement("button");
    button.className = `case-row ${row.status} ${row.case_id === state.activeCase ? "active" : ""}`;
    const title = document.createElement("strong");
    title.textContent = row.case_label;
    const meta = document.createElement("span");
    meta.textContent = `${row.site} · ${row.slide_count || 1} slide${row.slide_count > 1 ? "s" : ""}`;
    button.replaceChildren(title, meta);
    if (row.slide_review_required) {
      button.classList.add("needs-slide-review");
      const badge = document.createElement("span");
      badge.className = "slide-review-badge";
      badge.textContent = "Additional slides: review pending";
      button.append(badge);
    }
    button.onclick = () => openCase(row.case_id);
    return button;
  }));
  const done = state.cases.filter((row) => row.status === "submitted").length;
  const pending = state.cases.filter((row) => row.slide_review_required).length;
  $("progress").textContent = `${done} / ${state.cases.length}${pending ? ` · ${pending} recheck` : ""}`;
}

function makeReportCard(report) {
  const card = $("reportTemplate").content.firstElementChild.cloneNode(true);
  card.dataset.label = report.label;
  card.dataset.acceptability = "";
  card.querySelector(".report-label").textContent = report.label;
  card.querySelectorAll("[data-accept]").forEach((button) => {
    button.onclick = () => {
      card.dataset.acceptability = button.dataset.accept;
      card.querySelectorAll("[data-accept]").forEach((item) => item.classList.toggle("active", item === button));
      markDirty();
    };
  });
  return card;
}
function renderReports(reports) {
  $("reportGrid").replaceChildren(...reports.map(makeReportCard));
}

function normalizeRankGroups(value) {
  if (!Array.isArray(value) || !value.length) return [];
  if (value[0] === "tie") return [[...labels]];
  if (Array.isArray(value[0])) {
    return value
      .map((group) => group.filter((label) => labels.includes(label)))
      .filter((group) => group.length);
  }
  return value.filter((label) => labels.includes(label)).map((label) => [label]);
}
function flattenRankGroups(value) {
  return normalizeRankGroups(value).flat();
}
function selectedRankCount(value) {
  return flattenRankGroups(value).length;
}

function createRankControl(key) {
  const control = document.createElement("div");
  control.className = "rank-control";
  control.dataset.rankKey = key;
  const prompt = document.createElement("span");
  prompt.className = "rank-prompt";
  prompt.textContent = "Rank: 1 best, 5 worst";
  const selects = document.createElement("div");
  selects.className = "rank-selects";
  for (const label of labels) {
    const item = document.createElement("label");
    item.className = "rank-select-item";
    item.append(document.createTextNode(label));
    const select = document.createElement("select");
    select.dataset.rankLabel = label;
    select.setAttribute("aria-label", `Rank Report ${label}`);
    select.append(new Option("–", ""));
    for (let rank = 1; rank <= 5; rank += 1) select.append(new Option(String(rank), String(rank)));
    select.onchange = () => {
      const buckets = new Map();
      state.rankValues[key] = {};
      control.querySelectorAll("select[data-rank-label]").forEach((rankSelect) => {
        state.rankValues[key][rankSelect.dataset.rankLabel] = rankSelect.value;
        if (rankSelect.value === "") return;
        const rank = Number(rankSelect.value);
        if (!buckets.has(rank)) buckets.set(rank, []);
        buckets.get(rank).push(rankSelect.dataset.rankLabel);
      });
      state.rankOrders[key] = [...buckets.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([, group]) => group);
      updateRankControl(control, true);
    };
    item.append(select);
    selects.append(item);
  }
  const tie = document.createElement("button");
  tie.type = "button";
  tie.className = "rank-tie";
  tie.textContent = "All equivalent";
  tie.onclick = () => {
    state.rankOrders[key] = [[...labels]];
    state.rankValues[key] = Object.fromEntries(labels.map(label => [label, "1"]));
    updateRankControl(control); markDirty();
  };
  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "rank-reset";
  reset.textContent = "Clear";
  reset.onclick = () => { state.rankOrders[key] = []; state.rankValues[key] = {}; updateRankControl(control); markDirty(); };
  const status = document.createElement("span");
  status.className = "rank-status";
  control.append(prompt, selects, tie, reset, status);
  updateRankControl(control);
  return control;
}
function updateRankControl(control, preserveSelections = false) {
  const groups = normalizeRankGroups(state.rankOrders[control.dataset.rankKey]);
  const allEquivalent = groups.length === 1 && groups[0].length === labels.length;
  const rankByLabel = state.rankValues[control.dataset.rankKey] || {};
  if (!preserveSelections) {
    control.querySelectorAll("select[data-rank-label]").forEach((select) => {
      select.value = rankByLabel[select.dataset.rankLabel] ?? "";
    });
  }
  control.querySelector(".rank-tie").classList.toggle("active", allEquivalent);
  const status = control.querySelector(".rank-status");
  const expression = groups.map((group) => group.join(" = ")).join(" > ");
  if (selectedRankCount(groups) === labels.length) status.textContent = `Best ${expression} Worst`;
  else status.textContent = groups.length ? `${expression} · choose next` : "";
}
function updateAllRankControls() {
  document.querySelectorAll(".rank-control").forEach((control) => updateRankControl(control));
  updateMissingHighlights();
}

function updateMissingHighlights() {
  let required = 0, optional = 0;
  document.querySelectorAll('.rank-control').forEach(control => {
    const key = control.dataset.rankKey;
    const isRequired = key === 'overall' || coreSections.has(key.slice('section:'.length));
    let count = 0;
    control.querySelectorAll('select[data-rank-label]').forEach(select => {
      const missing = select.value === '';
      const label = select.dataset.rankLabel;
      select.classList.toggle('unranked', missing);
      select.setAttribute('aria-label', `Rank Report ${label}${missing ? ' (not ranked)' : ''}`);
      select.classList.remove('reused-score');
      select.title = missing ? 'Not ranked' : '';
      const row = control.closest('.section-row');
      const column = row?.querySelector(`[data-report-label="${label}"]`);
      column?.classList.toggle('pending-rating', missing && isRequired);
      if (missing) count++;
    });
    if (isRequired) required += count; else optional += count;
    control.closest('.section-row')?.classList.toggle('has-unranked', count > 0);
    const jump = [...document.querySelectorAll('#sectionJumpBar button')].find(x => x.dataset.rankKey === key);
    jump?.classList.toggle('pending-section', count > 0);
    if (jump) jump.title = `${key.slice('section:'.length)}: ${count} unranked${isRequired ? '' : ' (optional)'}`;

  });
  document.querySelectorAll('.report-card').forEach(card => {
    const missing = !card.dataset.acceptability;
    card.classList.toggle('pending-acceptability', missing);
    if (missing) required++;
  });
  const summary = $('pendingSummary');
  if (summary && state.activeCase) {
    summary.hidden = false;
    summary.textContent = `${required} required ratings remaining · ${optional} optional ranks unfilled`;
    summary.classList.toggle('complete', required === 0);
  }
}
function findRankControl(key) {
  return [...document.querySelectorAll(".rank-control")]
    .find((control) => control.dataset.rankKey === key);
}

function renderSectionComparison(reports) {
  state.rankOrders = {};
  state.rankValues = {};
  const reportMaps = new Map(reports.map((report) => [
    report.label, new Map(report.sections.map((section) => [section.title, section.body])),
  ]));
  const available = new Set(reports.flatMap((report) => report.sections.map((section) => section.title)));
  const titles = [...sectionOrder.filter((title) => available.has(title)), ...[...available].filter((title) => !sectionOrder.includes(title)).sort()];
  const rows = titles.map((title) => {
    const row = document.createElement("article");
    row.className = `section-row ${coreSections.has(title) ? "rankable core" : "rankable optional"}`;
    const heading = document.createElement("div");
    heading.className = "section-row-heading";
    const titleBlock = document.createElement("div");
    const h3 = document.createElement("h3");
    h3.textContent = title;
    const hint = document.createElement("span");
    hint.textContent = coreSections.has(title) ? "Core domain · ranking required" : "Optional section ranking";
    titleBlock.append(h3, hint);
    heading.append(titleBlock);
    heading.append(createRankControl(`section:${title}`));
    const columns = document.createElement("div");
    columns.className = "section-columns";
    for (const report of reports) {
      const bodyText = reportMaps.get(report.label).get(title);
      const column = document.createElement("section");
      column.className = `section-report ${bodyText ? "" : "missing"}`;
      column.dataset.reportLabel = report.label;
      const label = document.createElement("strong");
      label.textContent = `Report ${report.label}`;
      const body = document.createElement("div");
      body.className = "section-body";
      body.textContent = bodyText || "This report does not provide a distinct section for this domain.";
      column.append(label, body);
      columns.append(column);
    }
    row.append(heading, columns);
    return row;
  });
  $("sectionComparison").replaceChildren(...rows);
  const jumpBar = $("sectionJumpBar");
  const count = document.createElement("strong");
  count.textContent = `${titles.length} sections`;
  const links = titles.map((title, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = title;
    button.dataset.rankKey = `section:${title}`;
    button.title = `Jump to ${title}`;
    button.onclick = () => rows[index].scrollIntoView({behavior: "smooth", block: "start"});
    return button;
  });
  jumpBar.replaceChildren(count, ...links);
  $("overallRankControl").replaceChildren(createRankControl("overall"));
}

function rankingMapToOrder(values) {
  if (!values || typeof values !== "object") return [];
  if (labels.every((label) => values[label] === "tie")) return [[...labels]];
  const entries = Object.entries(values)
    .filter(([label, value]) => labels.includes(label) && /^[1-5]$/.test(String(value)))
    .sort((a, b) => Number(a[1]) - Number(b[1]) || labels.indexOf(a[0]) - labels.indexOf(b[0]));
  const groups = [];
  for (const [label, rank] of entries) {
    const previous = groups.at(-1);
    if (previous?.rank === String(rank)) previous.labels.push(label);
    else groups.push({rank: String(rank), labels: [label]});
  }
  return groups.map((group) => group.labels);
}
function fillAnnotation(annotation) {
  state.rankOrders = {};
  state.rankValues = {};
  const ratings = annotation?.ratings || {};
  document.querySelectorAll(".report-card").forEach((card) => {
    const values = ratings[card.dataset.label] || {};
    card.dataset.acceptability = values.acceptability || "";
    card.querySelectorAll("[data-accept]").forEach((button) => button.classList.toggle("active", button.dataset.accept === values.acceptability));
    card.querySelector('[data-field="unsupported_major_claim"]').checked = values.unsupported_major_claim === "yes" || values.unsupported_major_claim === true;
    card.querySelector('[data-field="major_visible_omission"]').checked = values.major_visible_omission === "yes" || values.major_visible_omission === true;
  });
  for (const [section, values] of Object.entries(annotation?.section_rankings || {})) {
    if (findRankControl(`section:${section}`)) {
      state.rankValues[`section:${section}`] = Object.fromEntries(labels.map(label =>
        [label, values[label] === "tie" ? "1" : String(values[label] ?? "")]));
      state.rankOrders[`section:${section}`] = rankingMapToOrder(values);
    }
  }
  if (annotation?.overall_rank_values) {
    state.rankValues.overall = Object.fromEntries(labels.map(label => [label, String(annotation.overall_rank_values[label] ?? "")]));
    state.rankOrders.overall = rankingMapToOrder(state.rankValues.overall);
  }
  else if (Array.isArray(annotation?.overall_rank_groups)) state.rankOrders.overall = normalizeRankGroups(annotation.overall_rank_groups);
  else if (Array.isArray(annotation?.overall_ranking)) state.rankOrders.overall = normalizeRankGroups(annotation.overall_ranking);
  else if (annotation?.best_report_tie) state.rankOrders.overall = [[...labels]];
  else if (annotation?.best_report) state.rankOrders.overall = [[annotation.best_report]];
  if (!state.rankValues.overall) state.rankValues.overall = orderToRanking(state.rankOrders.overall);
  updateAllRankControls();
  const notes = $("notes");
  if (notes) notes.value = annotation?.comparison_notes || "";
  $("saveStatus").textContent = annotation?.status ? `Loaded ${annotation.status}` : "Not saved";
  // An explicit draft uncheck overrides the previous submitted confirmation.
  const hasDraftConfirmation = typeof annotation?.confirm_slide_review === "boolean";
  $("confirmSlideReview").checked = Boolean(
    state.meta?.slide_set_revision && (hasDraftConfirmation
      ? annotation.confirm_slide_review && annotation.slide_set_revision === state.meta.slide_set_revision
      : annotation?.slide_review?.confirmed_revision === state.meta.slide_set_revision
    ),
  );
}

async function openCase(caseId) {
  if (!state.reader) return alert("Start a reader session first.");
  const requestSeq = ++state.caseRequestSeq;
  try {
    if (state.dirty) writeLocalDraft();
    const meta = await api(withReader(`/api/case/${encodeURIComponent(caseId)}`));
    if (requestSeq !== state.caseRequestSeq) return;
    state.activeCase = caseId;
    state.meta = meta;
    state.activeSlideId = null;
    state.meta.slides = meta.slides?.length ? meta.slides : [
      {slide_id: "S001", label: "Slide 1", dzi_url: `/api/dzi/${encodeURIComponent(caseId)}.dzi`},
    ];
    $("slideSelect").replaceChildren(...state.meta.slides.map((slide, index) =>
      new Option(`${slide.label} / ${state.meta.slides.length}`, slide.slide_id)));
    $("slideSelect").disabled = state.meta.slides.length < 2;
    $("slideReviewConfirmation").hidden = state.meta.slides.length < 2;
    updateSlideReviewNotice();
    $("caseTitle").textContent = state.meta.case_label;
    $("caseMeta").textContent = `${state.meta.site} · five candidate reports`;
    renderSectionComparison(state.meta.reports);
    renderReports(state.meta.reports);
    fillAnnotation(state.meta.annotation);
    const localDraft = readLocalDraft();
    state.dirty = Boolean(localDraft);
    if (localDraft) {
      fillAnnotation(localDraft.payload);
      setSyncStatus("Recovered unsynced draft from this device", "local", "Local draft recovered");
      if (!localDraft.server_was_submitted) scheduleAutosave(100);
    } else {
      setSyncStatus(
        state.meta.annotation?.status ? `Loaded ${state.meta.annotation.status}` : "Not saved",
        state.meta.annotation?.status ? "synced" : "idle",
        state.meta.annotation?.status ? "Synced" : "Ready",
      );
    }
    selectPatientSlide(state.meta.slides[0].slide_id);
    renderCases();
  } catch (error) { alert(error.message); }
}

function updateSlideReviewNotice() {
  $("slideReviewNotice").hidden = !state.meta?.slide_review_required;
}

function selectPatientSlide(slideId, publish = true) {
  const slide = state.meta?.slides?.find(item => item.slide_id === slideId);
  if (!slide || slideId === state.activeSlideId) return;
  state.activeSlideId = slideId;
  $("slideSelect").value = slideId;
  $("slideLoadStatus").textContent = `${slide.label}: loading`;
  // Switching evidence must never recreate reports, ranking controls or drafts.
  state.viewer.open(slide.dzi_url);
  if (publish) publishViewerWindowCase();
}

function receiveSlideSelection(message) {
  if (message?.type === "select-slide" && message.case_id === state.activeCase) {
    selectPatientSlide(message.slide_id);
  }
}

function viewerWindowCaseMessage() {
  if (!state.activeCase) return null;
  return {
    type: "open-case",
    case_id: state.activeCase,
    site: state.meta?.site || "not provided",
    slide_id: state.activeSlideId,
    slides: state.meta?.slides || [],
  };
}

function publishViewerWindowCase() {
  const message = viewerWindowCaseMessage();
  if (!message) return;
  localStorage.setItem(viewerWindowStorageKey, JSON.stringify(message));
  viewerWindowChannel?.postMessage(message);
}

function openViewerWindow() {
  if (!state.activeCase) return alert("Select a case first.");
  publishViewerWindowCase();
  const name = `pathxai_wsi_viewer_${location.port || "default"}`;
  state.viewerWindow = window.open(
    "/static/viewer_window.html",
    name,
    "popup,width=1600,height=1000,resizable=yes,scrollbars=no",
  );
  if (!state.viewerWindow) {
    alert("The WSI window was blocked. Allow pop-ups for this site and try again.");
    return;
  }
  state.viewerWindow.focus();
  setTimeout(publishViewerWindowCase, 250);
}

async function toggleViewerFullscreen() {
  const panel = document.querySelector(".viewer-card");
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await panel.requestFullscreen();
  } catch (error) {
    alert(`Full-screen mode is unavailable: ${error.message}`);
  }
}

function updateViewerFullscreenButton() {
  $("toggleViewerFullscreen").textContent = document.fullscreenElement ? "Exit full screen" : "Full screen";
  resizeViewer();
}

function orderToRanking(order) {
  const groups = normalizeRankGroups(order);
  const ranking = Object.fromEntries(labels.map((label) => [label, ""]));
  groups.forEach((group, index) => {
    group.forEach((label) => { ranking[label] = String(index + 1); });
  });
  return ranking;
}
function collectRatings() {
  const ratings = {};
  document.querySelectorAll(".report-card").forEach((card) => {
    ratings[card.dataset.label] = {
      acceptability: card.dataset.acceptability || "",
      unsupported_major_claim: card.querySelector('[data-field="unsupported_major_claim"]').checked ? "yes" : "no",
      major_visible_omission: card.querySelector('[data-field="major_visible_omission"]').checked ? "yes" : "no",
    };
  });
  return ratings;
}
function collectSectionRankings() {
  const rankings = {};
  document.querySelectorAll('[data-rank-key^="section:"]').forEach((control) => {
    const section = control.dataset.rankKey.slice("section:".length);
    const order = state.rankOrders[control.dataset.rankKey] || [];
    if (order.length) rankings[section] = {...state.rankValues[control.dataset.rankKey]};
  });
  return rankings;
}

function buildPayload(status = "draft") {
  const overallGroups = normalizeRankGroups(state.rankOrders.overall);
  const overall = overallGroups.flat();
  const topGroup = overallGroups[0] || [];
  return {
    reader: state.reader,
    rank_scale: {...rankScale},
    base_revision: state.meta?.annotation?.revision ?? 0,
    status,
    ratings: collectRatings(),
    section_rankings: collectSectionRankings(),
    overall_ranking: overall,
    overall_rank_groups: overallGroups,
    overall_rank_values: {...state.rankValues.overall},
    best_report: topGroup.length === 1 ? topGroup[0] : null,
    best_report_tie: topGroup.length > 1,
    comparison_notes: $("notes")?.value || "",
    slide_set_revision: state.meta?.slide_set_revision,
    confirm_slide_review: $("confirmSlideReview").checked,
  };
}

function localDraftKey(caseId = state.activeCase, readerId = state.reader?.reader_id) {
  if (!caseId || !readerId) return null;
  return `${activeDraftPrefix()}:${encodeURIComponent(readerId)}:${encodeURIComponent(caseId)}`;
}

function activeDraftPrefix() {
  return draftStoragePrefix;
}

function setSyncStatus(message, syncState = "idle", headerMessage = null) {
  const saveStatus = $("saveStatus");
  if (saveStatus) {
    saveStatus.textContent = message;
    saveStatus.dataset.syncState = syncState;
  }
  const connectionStatus = $("connectionStatus");
  if (connectionStatus) {
    connectionStatus.textContent = headerMessage || message;
    connectionStatus.dataset.syncState = syncState;
  }
}

function readLocalDraft(caseId = state.activeCase, readerId = state.reader?.reader_id) {
  const key = localDraftKey(caseId, readerId);
  if (!key) return null;
  try {
    const value = JSON.parse(localStorage.getItem(key));
    return value?.payload || null;
  } catch (_) {
    localStorage.removeItem(key);
    return null;
  }
}

function writeLocalDraft() {
  const key = localDraftKey();
  if (!key) return null;
  const record = {
    schema_version: "pathxai_browser_draft_v2",
    saved_at: new Date().toISOString(),
    reader_id: state.reader.reader_id,
    case_id: state.activeCase,
    server_was_submitted: state.meta?.annotation?.status === "submitted",
    payload: buildPayload("draft"),
  };
  localStorage.setItem(key, JSON.stringify(record));
  return record;
}

function clearLocalDraft(caseId, readerId, savedAt = null) {
  const key = localDraftKey(caseId, readerId);
  if (!key) return;
  if (savedAt) {
    const current = readLocalDraft(caseId, readerId);
    if (current && current.saved_at !== savedAt) return;
  }
  localStorage.removeItem(key);
}

function pendingLocalDrafts() {
  const readerId = state.reader?.reader_id;
  if (!readerId) return [];
  const records = [];
  for (let index = 0; index < localStorage.length; index += 1) {
    const key = localStorage.key(index);
    if (!key?.startsWith(`${activeDraftPrefix()}:${encodeURIComponent(readerId)}:`)) continue;
    try {
      const record = JSON.parse(localStorage.getItem(key));
      if (record?.reader_id === readerId && record.payload && !record.server_was_submitted) records.push(record);
    } catch (_) {
      // Leave unrelated or malformed browser storage untouched.
    }
  }
  return records.sort((a, b) => String(a.saved_at).localeCompare(String(b.saved_at)));
}

function scheduleAutosave(delay = autosaveDelayMs) {
  clearTimeout(state.autosaveTimer);
  state.autosaveTimer = setTimeout(flushAutosave, delay);
}

function scheduleAutosaveRetry() {
  clearTimeout(state.autosaveRetryTimer);
  state.autosaveRetryTimer = setTimeout(flushAutosave, autosaveRetryMs);
}

async function waitForAutosaveIdle() {
  while (state.autosaveInFlight) {
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}

function markDirty() {
  if (!state.activeCase || !state.reader?.reader_id) return;
  updateMissingHighlights();
  state.dirty = true;
  const record = writeLocalDraft();
  if (!record) return;
  if (record.server_was_submitted) {
    setSyncStatus("Saved on this device; submit to update the reviewed case", "local", "Local draft safe");
    return;
  }
  setSyncStatus(
    navigator.onLine ? "Saved on this device; syncing..." : "Offline; saved on this device; will retry",
    navigator.onLine ? "syncing" : "offline",
    navigator.onLine ? "Syncing..." : "Offline; draft safe",
  );
  scheduleAutosave();
}

async function flushAutosave() {
  clearTimeout(state.autosaveTimer);
  clearTimeout(state.autosaveRetryTimer);
  if (state.autosaveInFlight || !state.reader?.reader_id) return;
  const record = pendingLocalDrafts()[0];
  if (!record) return;
  state.autosaveInFlight = true;
  try {
    const path = `/api/annotation/${encodeURIComponent(record.case_id)}?reader_id=${encodeURIComponent(record.reader_id)}`;
    const saved = await api(path, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({...record.payload, status: "draft", confirm_edit_submitted: false}),
    });
    clearLocalDraft(record.case_id, record.reader_id, record.saved_at);
    if (state.activeCase === record.case_id && state.reader?.reader_id === record.reader_id) {
      const newer = readLocalDraft(record.case_id, record.reader_id);
      if (newer) {
        newer.payload.base_revision = saved.revision;
        localStorage.setItem(localDraftKey(record.case_id, record.reader_id), JSON.stringify(newer));
      }
      state.meta.annotation = saved;
      state.meta.slide_review_required = saved.slide_review_required;
      updateSlideReviewNotice();
      state.dirty = Boolean(newer);
      if (!newer) {
        fillAnnotation(saved);
        setSyncStatus(`Draft synced ${new Date().toLocaleTimeString()}`, "synced", "Synced");
      }
      const row = state.cases.find(item => item.case_id === record.case_id);
      if (row) { row.status = saved.status; row.slide_review_required = saved.slide_review_required; }
      renderCases();
    }
    if (pendingLocalDrafts().length) scheduleAutosave(50);
  } catch (error) {
    if (state.activeCase === record.case_id) {
      setSyncStatus(error.status === 400 || error.status === 409
        ? `${error.message} Local draft retained.` : "Offline; saved on this device; will retry",
        error.status ? "local" : "offline", error.status ? "Review save conflict" : "Offline; draft safe");
    }
    if (error.status !== 400 && error.status !== 409) scheduleAutosaveRetry();
  } finally {
    state.autosaveInFlight = false;
  }
}
function rankingComplete(order) {
  return selectedRankCount(order) === labels.length;
}
function validateSubmission(payload) {
  if (state.meta?.slides?.length > 1 && !payload.confirm_slide_review) {
    throw new Error("Confirm that you have reviewed all available slides before submitting this patient case.");
  }
  for (const [label, values] of Object.entries(payload.ratings)) {
    if (!values.acceptability) throw new Error(`Select clinical acceptability for Report ${label}.`);
  }
  for (const section of coreSections) {
    const control = findRankControl(`section:${section}`);
    if (control && !rankingComplete(state.rankOrders[`section:${section}`])) {
      throw new Error(`Complete the required A-E ranking for ${section}.`);
    }
  }
  for (const [key, order] of Object.entries(state.rankOrders)) {
    if (key.startsWith("section:") && selectedRankCount(order) && !rankingComplete(order)) {
      throw new Error(`Finish or clear the optional A-E ranking for ${key.slice("section:".length)}.`);
    }
  }
  if (!rankingComplete(state.rankOrders.overall)) throw new Error("Complete the overall report order or select All equivalent.");
}
function deriveOverallRanking() {
  const completed = [...document.querySelectorAll('[data-rank-key^="section:"]')]
    .map((control) => state.rankOrders[control.dataset.rankKey])
    .filter(rankingComplete);
  if (!completed.length) return alert("Complete at least one section ranking first.");
  const scores = Object.fromEntries(labels.map((label) => [label, 0]));
  for (const value of completed) {
    const groups = normalizeRankGroups(value);
    let position = 1;
    for (const group of groups) {
      const averagePosition = position + (group.length - 1) / 2;
      group.forEach((label) => { scores[label] += averagePosition; });
      position += group.length;
    }
  }
  const sorted = [...labels].sort((a, b) => scores[a] - scores[b] || a.localeCompare(b));
  const groups = [];
  for (const label of sorted) {
    const previous = groups.at(-1);
    if (previous && Math.abs(previous.score - scores[label]) < 1e-9) previous.labels.push(label);
    else groups.push({score: scores[label], labels: [label]});
  }
  state.rankOrders.overall = groups.map((group) => group.labels);
  state.rankValues.overall = orderToRanking(state.rankOrders.overall);
  updateAllRankControls();
  markDirty();
}
async function save(status) {
  if (!state.activeCase) return alert("Select a case first.");
  const existingSubmitted = state.meta?.annotation?.status === "submitted";
  if (existingSubmitted && !confirm("This review was submitted. Save a revised version and retain the old version in history?")) return;
  if (status === "submitted") {
    try {
      validateSubmission(buildPayload(status));
    } catch (error) {
      markDirty();
      return alert(error.message);
    }
    if (!confirm("Submit this comparison? You can still revise it later after an extra confirmation.")) return;
  }
  clearTimeout(state.autosaveTimer);
  clearTimeout(state.autosaveRetryTimer);
  await waitForAutosaveIdle();
  clearTimeout(state.autosaveTimer);
  const localRecord = writeLocalDraft();
  const payload = {...buildPayload(status), confirm_edit_submitted: existingSubmitted};
  try {
    const saved = await api(withReader(`/api/annotation/${encodeURIComponent(state.activeCase)}`), {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
    });
    clearLocalDraft(state.activeCase, state.reader.reader_id);
    state.dirty = false;
    state.meta.annotation = saved;
    state.meta.slide_review_required = saved.slide_review_required;
    updateSlideReviewNotice();
    const row = state.cases.find((item) => item.case_id === state.activeCase);
    if (row) { row.status = saved.status; row.slide_review_required = saved.slide_review_required; }
    setSyncStatus(`Saved ${saved.status}`, "synced", "Synced");
    renderCases();
  } catch (error) {
    if (!localRecord) writeLocalDraft();
    setSyncStatus("Save failed; draft remains safe on this device", "offline", "Offline; draft safe");
    scheduleAutosaveRetry();
    alert(`${error.message}\nYour draft is saved on this device and will retry automatically.`);
  }
}

function resizeViewer() {
  setTimeout(() => {
    state.viewer?.viewport?.resize();
    state.viewer?.forceRedraw();
  }, 80);
}
function updateViewerDock() {
  const dock = $("viewerDock");
  dock.classList.toggle("viewer-pinned", state.viewerPinned);
  dock.classList.toggle("viewer-collapsed", state.viewerCollapsed);
  $("pinViewer").textContent = state.viewerPinned ? "Unpin" : "Pin";
  $("collapseViewer").textContent = state.viewerCollapsed ? "Expand" : "Collapse";
  state.viewer?.setMouseNavEnabled(!state.viewerCollapsed);
  resizeViewer();
}
function initViewer() {
  state.viewer = OpenSeadragon({
    id: "wsiViewer", prefixUrl: "", showNavigator: true, navigatorPosition: "BOTTOM_RIGHT",
    showNavigationControl: false, maxZoomPixelRatio: 2.5, visibilityRatio: 0.5,
  });
  state.viewer.addHandler("open", () => { $("slideLoadStatus").textContent = "Ready"; });
  state.viewer.addHandler("open-failed", () => { $("slideLoadStatus").textContent = "Slide failed to load"; });
  $("slideSelect").onchange = () => selectPatientSlide($("slideSelect").value);
  $("zoomIn").onclick = () => state.viewer.viewport.zoomBy(1.5);
  $("zoomOut").onclick = () => state.viewer.viewport.zoomBy(0.67);
  $("resetView").onclick = () => state.viewer.viewport.goHome();
  $("openViewerWindow").onclick = openViewerWindow;
  $("toggleViewerFullscreen").onclick = toggleViewerFullscreen;
  $("pinViewer").onclick = () => { state.viewerPinned = !state.viewerPinned; updateViewerDock(); };
  $("collapseViewer").onclick = () => { state.viewerCollapsed = !state.viewerCollapsed; updateViewerDock(); };
}
function restoreReader() {
  try {
    const profile = JSON.parse(localStorage.getItem(readerKey));
    if (!profile?.reader_id) return;
    state.reader = profile;
    $("readerId").value = profile.reader_id;
    $("readerStatus").textContent = `Active: ${profile.reader_id}`;
    loadCases().catch((error) => alert(error.message));
  } catch (_) { localStorage.removeItem(readerKey); }
}

$("startReader").onclick = startReader;
$("saveDraft").onclick = () => save("draft");
$("submitReview").onclick = () => save("submitted");
$("deriveOverall").onclick = deriveOverallRanking;
$("reportGrid").addEventListener("change", markDirty);
$("sectionComparison").addEventListener("change", markDirty);
$("overallRankControl").addEventListener("change", markDirty);
$("notes").addEventListener("input", markDirty);
$("confirmSlideReview").addEventListener("change", markDirty);
$("caseSearch").oninput = renderCases;
document.querySelectorAll(".case-filters button").forEach((button) => button.onclick = () => {
  document.querySelectorAll(".case-filters button").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  state.filter = button.dataset.filter;
  renderCases();
});
window.addEventListener("online", () => {
  setSyncStatus("Connection restored; syncing local drafts...", "syncing", "Reconnected; syncing...");
  scheduleAutosave(50);
});
window.addEventListener("offline", () => {
  if (state.dirty || pendingLocalDrafts().length) {
    setSyncStatus("Offline; saved on this device; will retry", "offline", "Offline; draft safe");
  } else {
    setSyncStatus("Offline", "offline", "Offline");
  }
});
window.addEventListener("beforeunload", () => {
  if (state.dirty) writeLocalDraft();
});
viewerWindowChannel?.addEventListener("message", (event) => {
  if (event.data?.type === "viewer-ready") publishViewerWindowCase();
  receiveSlideSelection(event.data);
});
window.addEventListener("storage", (event) => {
  if (event.key === viewerWindowSelectionKey && event.newValue) {
    try { receiveSlideSelection(JSON.parse(event.newValue)); } catch (_) { /* Ignore malformed state. */ }
  }
});
document.addEventListener("fullscreenchange", updateViewerFullscreenButton);
initViewer();
restoreReader();
