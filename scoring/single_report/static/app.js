const state = {
  cases: [],
  activeCase: null,
  meta: null,
  filter: "all",
  viewer: null,
  diagnosticRois: [],
  roiOverlays: new Map(),
  roiTracker: null,
  roiDrawing: false,
  roiStart: null,
  roiDraftElement: null,
  viewerWindow: null,
  readerProfile: null,
  reportLocked: false,
  editingSubmitted: false,
  dirty: false,
  autosaveTimer: null,
  autosaveRetryTimer: null,
  autosaveInFlight: false,
};
const viewerWindowChannelName = "pathxai-wsi-viewer-v1";
const viewerWindowStorageKey = "pathxai-wsi-viewer-case-v1";
const viewerWindowChannel = "BroadcastChannel" in window
  ? new BroadcastChannel(viewerWindowChannelName)
  : null;

const readerStorageKey = "pathxai_tutorial_single_report_session_v1";
const draftStoragePrefix = "pathxai_tutorial_single_report_draft_v1";
const autosaveDelayMs = 1200;
const autosaveRetryMs = 10000;

const roiTypeLabels = {
  diagnostic: "Diagnostic",
  invasion: "Invasion / interface",
  grade: "Grade / atypia",
  necrosis: "Necrosis",
  other: "Other",
};

const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), opts.timeoutMs || 20000);
  try {
    const res = await fetch(path, {...opts, signal: controller.signal});
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json();
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Request timed out; the local draft is safe.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function formIds() {
  return [
    "readerId",
    "primaryDiagnosis", "broadType", "histologicSubtype",
    "malignancyStatus", "invasiveStatus", "grade", "necrosis", "confidence",
    "notAssessable", "sectionDiagnosis", "keyFindings", "evidenceRationale",
    "limitations", "finalSummary", "diagnosticCorrectness", "unsupportedClaims", "notes"
  ];
}

function editableReportIds() {
  return formIds().filter(id => ![
    "readerId"
  ].includes(id));
}

function setReportLocked(locked) {
  state.reportLocked = Boolean(locked);
  editableReportIds().forEach(id => { $(id).disabled = state.reportLocked; });
  ["roiType", "roiNote", "drawRoi", "saveDraft"].forEach(id => {
    $(id).disabled = state.reportLocked;
  });
  $("submitReport").disabled = false;
  $("submitReport").textContent = state.reportLocked
    ? "Edit Submitted Report"
    : state.editingSubmitted
      ? "Re-submit / Lock"
      : "Submit / Lock";
  if (state.reportLocked) setRoiDrawing(false);
  renderRoiList();
}

function unlockSubmittedForEditing() {
  if (!state.reportLocked) return;
  const confirmed = confirm(
    "This report has already been submitted. Unlock it for correction? " +
    "The submitted version will remain in the audit history."
  );
  if (!confirmed) return;
  state.editingSubmitted = true;
  setReportLocked(false);
  $("saveStatus").textContent = "Editing submitted report · not yet re-saved";
}

function getVal(id) { return $(id).value.trim(); }
function setVal(id, val) { $(id).value = val ?? ""; }

function readerApiPath(path) {
  if (!state.readerProfile?.reader_id) throw new Error("No active reader session");
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}reader_id=${encodeURIComponent(state.readerProfile.reader_id)}`;
}

function collectAnnotation(status = "draft") {
  return {
    status,
    reader: {...state.readerProfile},
    structured_facts: {
      primary_diagnosis: getVal("primaryDiagnosis"),
      broad_type: getVal("broadType"),
      histologic_subtype: getVal("histologicSubtype"),
      malignancy_status: getVal("malignancyStatus"),
      invasive_status: getVal("invasiveStatus"),
      grade: getVal("grade"),
      necrosis: getVal("necrosis"),
      confidence: getVal("confidence"),
      not_assessable: getVal("notAssessable").split(",").map(s => s.trim()).filter(Boolean),
    },
    narrative_report: {
      diagnosis: getVal("sectionDiagnosis"),
      key_microscopic_findings: getVal("keyFindings"),
      evidence_rationale: getVal("evidenceRationale"),
      limitations: getVal("limitations"),
      final_summary: getVal("finalSummary"),
    },
    diagnostic_rois: state.diagnosticRois,
    review_scores: {
      diagnostic_correctness: getVal("diagnosticCorrectness"),
      unsupported_claims: getVal("unsupportedClaims"),
    },
    notes: getVal("notes"),
  };
}

function localDraftKey(caseId = state.activeCase, readerId = state.readerProfile?.reader_id) {
  if (!caseId || !readerId) return null;
  return `${draftStoragePrefix}:${encodeURIComponent(readerId)}:${encodeURIComponent(caseId)}`;
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

function readLocalDraft(caseId = state.activeCase, readerId = state.readerProfile?.reader_id) {
  const key = localDraftKey(caseId, readerId);
  if (!key) return null;
  try {
    const value = JSON.parse(localStorage.getItem(key));
    return value?.payload ? value : null;
  } catch (_) {
    localStorage.removeItem(key);
    return null;
  }
}

function writeLocalDraft() {
  const key = localDraftKey();
  if (!key || state.reportLocked) return null;
  const record = {
    schema_version: "pathxai_browser_draft_v2",
    saved_at: new Date().toISOString(),
    reader_id: state.readerProfile.reader_id,
    case_id: state.activeCase,
    server_was_submitted: state.editingSubmitted || state.meta?.annotation?.status === "submitted",
    payload: collectAnnotation("draft"),
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
  const readerId = state.readerProfile?.reader_id;
  if (!readerId) return [];
  const records = [];
  for (let index = 0; index < localStorage.length; index += 1) {
    const key = localStorage.key(index);
    if (!key?.startsWith(`${draftStoragePrefix}:`)) continue;
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
  if (!state.activeCase || !state.readerProfile?.reader_id || state.reportLocked) return;
  state.dirty = true;
  const record = writeLocalDraft();
  if (!record) return;
  if (record.server_was_submitted) {
    setSyncStatus("Saved on this device; re-submit to update the locked report", "local", "Local draft safe");
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
  if (state.autosaveInFlight || !state.readerProfile?.reader_id) return;
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
    if (state.activeCase === record.case_id && state.readerProfile?.reader_id === record.reader_id) {
      const newer = readLocalDraft(record.case_id, record.reader_id);
      state.meta.annotation = saved;
      state.dirty = Boolean(newer);
      if (!newer) {
        setSyncStatus(`Draft synced ${new Date().toLocaleTimeString()}`, "synced", "Synced");
      }
      const row = state.cases.find(c => c.case_id === record.case_id);
      if (row) {
        row.annotation_status = saved.status;
        row.updated_at = saved.updated_at;
      }
      renderCases();
    }
    if (pendingLocalDrafts().length) scheduleAutosave(50);
  } catch (error) {
    if (state.activeCase === record.case_id) {
      setSyncStatus("Offline; saved on this device; will retry", "offline", "Offline; draft safe");
    }
    scheduleAutosaveRetry();
  } finally {
    state.autosaveInFlight = false;
  }
}

function fillAnnotation(ann) {
  for (const id of formIds()) setVal(id, "");
  state.diagnosticRois = Array.isArray(ann?.diagnostic_rois)
    ? ann.diagnostic_rois.map(roi => ({...roi}))
    : [];
  renderRoiList();
  renderRoiOverlays();
  setVal("readerId", state.readerProfile?.reader_id);
  if (!ann) return;
  const sf = ann.structured_facts || {};
  setVal("primaryDiagnosis", sf.primary_diagnosis);
  setVal("broadType", sf.broad_type);
  setVal("histologicSubtype", sf.histologic_subtype);
  setVal("malignancyStatus", sf.malignancy_status);
  setVal("invasiveStatus", sf.invasive_status);
  setVal("grade", sf.grade);
  setVal("necrosis", sf.necrosis);
  setVal("confidence", sf.confidence);
  setVal("notAssessable", (sf.not_assessable || []).join(", "));
  const nr = ann.narrative_report || {};
  setVal("sectionDiagnosis", nr.diagnosis);
  setVal("keyFindings", nr.key_microscopic_findings);
  setVal("evidenceRationale", nr.evidence_rationale);
  setVal("limitations", nr.limitations);
  setVal("finalSummary", nr.final_summary);
  const rs = ann.review_scores || {};
  setVal("diagnosticCorrectness", rs.diagnostic_correctness);
  setVal("unsupportedClaims", rs.unsupported_claims);
  setVal("notes", ann.notes);
}

function renderCases() {
  const q = $("caseSearch").value.toLowerCase();
  const list = $("caseList");
  list.innerHTML = "";
  const rows = state.cases.filter(c => {
    const status = c.annotation_status || "not_started";
    const text = `${c.case_label || ""} ${c.site || ""}`.toLowerCase();
    return (state.filter === "all" || status === state.filter) && text.includes(q);
  });
  for (const c of rows) {
    const div = document.createElement("div");
    const status = c.annotation_status || "not_started";
    div.className = `case-card ${state.activeCase === c.case_id ? "active" : ""}`;
    div.innerHTML = `
      <strong>${c.case_label || "Case"}</strong>
      <span>Known site: ${c.site || "not provided"}</span><br>
      <span class="status-pill ${status}">${status}</span>
    `;
    div.onclick = () => loadCase(c.case_id);
    list.appendChild(div);
  }
}

async function loadCases() {
  if (!state.readerProfile?.reader_id) return;
  const status = await api("/api/status");
  $("statusLine").textContent = `${status.case_count} cases · active local session ${state.readerProfile.reader_id}`;
  state.cases = await api(readerApiPath("/api/cases"));
  renderCases();
  if (!state.activeCase && state.cases.length) {
    await loadCase(state.cases[0].case_id);
  }
}

async function loadCase(caseId) {
  if (state.dirty) writeLocalDraft();
  const meta = await api(readerApiPath(`/api/case/${caseId}`));
  setRoiDrawing(false);
  state.viewer.close();
  removeRoiOverlays();
  state.activeCase = caseId;
  state.meta = meta;
  $("caseTitle").textContent = state.meta.case_label || "Case";
  $("caseMeta").textContent = `Known site: ${state.meta.site || "not provided"}`;
  fillAnnotation(state.meta.annotation);
  const localDraft = readLocalDraft();
  const submitted = state.meta.annotation?.status === "submitted";
  state.editingSubmitted = Boolean(localDraft && (submitted || localDraft.server_was_submitted));
  state.dirty = Boolean(localDraft);
  if (localDraft) {
    fillAnnotation(localDraft.payload);
    setReportLocked(false);
    setSyncStatus("Recovered unsynced draft from this device", "local", "Local draft recovered");
    if (!localDraft.server_was_submitted) scheduleAutosave(100);
  } else {
    setReportLocked(submitted);
    setSyncStatus(
      submitted ? "Submitted and locked" : state.meta.annotation?.status ? `Loaded ${state.meta.annotation.status}` : "No draft yet",
      submitted || state.meta.annotation?.status ? "synced" : "idle",
      submitted || state.meta.annotation?.status ? "Synced" : "Ready",
    );
  }
  renderCases();
  $("viewerPlaceholder")?.remove();
  $("viewerScale").textContent = "Loading slide…";
  state.viewer.open(`/api/dzi/${encodeURIComponent(caseId)}.dzi`);
  publishViewerWindowCase();
}

function viewerWindowCaseMessage() {
  if (!state.activeCase) return null;
  return {
    type: "open-case",
    case_id: state.activeCase,
    site: state.meta?.site || "not provided",
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
  const panel = document.querySelector(".viewer-panel");
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await panel.requestFullscreen();
  } catch (error) {
    alert(`Full-screen mode is unavailable: ${error.message}`);
  }
}

function updateViewerFullscreenButton() {
  const button = $("toggleViewerFullscreen");
  if (button) button.textContent = document.fullscreenElement ? "Exit full screen" : "Full screen";
  setTimeout(() => {
    state.viewer?.viewport?.resize();
    state.viewer?.forceRedraw();
  }, 50);
}

function roiClass(type) {
  return `roi-${Object.prototype.hasOwnProperty.call(roiTypeLabels, type) ? type : "other"}`;
}

function removeRoiOverlays() {
  for (const element of state.roiOverlays.values()) {
    state.viewer.removeOverlay(element);
  }
  state.roiOverlays.clear();
}

function makeRoiOverlay(roi, draft = false) {
  const element = document.createElement("div");
  element.className = `diagnostic-roi-overlay ${roiClass(roi.type)}${draft ? " drawing" : ""}`;
  if (!draft) {
    const label = document.createElement("span");
    label.textContent = roiTypeLabels[roi.type] || roiTypeLabels.other;
    element.appendChild(label);
  }
  return element;
}

function renderRoiOverlays() {
  if (!state.viewer || !state.viewer.world.getItemCount()) return;
  removeRoiOverlays();
  const item = state.viewer.world.getItemAt(0);
  for (const roi of state.diagnosticRois) {
    if (![roi.x, roi.y, roi.width, roi.height].every(Number.isFinite)) continue;
    const element = makeRoiOverlay(roi);
    const rect = item.imageToViewportRectangle(roi.x, roi.y, roi.width, roi.height);
    state.viewer.addOverlay({element, location: rect, checkResize: false});
    state.roiOverlays.set(roi.id, element);
  }
}

function focusRoi(roi) {
  if (!state.viewer.world.getItemCount()) return;
  const item = state.viewer.world.getItemAt(0);
  const rect = item.imageToViewportRectangle(roi.x, roi.y, roi.width, roi.height);
  state.viewer.viewport.fitBounds(rect, false);
}

function removeRoi(roiId) {
  if (state.reportLocked) return;
  state.diagnosticRois = state.diagnosticRois.filter(roi => roi.id !== roiId);
  renderRoiList();
  renderRoiOverlays();
  markDirty();
}

function renderRoiList() {
  const list = $("roiList");
  if (!list) return;
  list.innerHTML = "";
  $("roiCount").textContent = `${state.diagnosticRois.length} ${state.diagnosticRois.length === 1 ? "region" : "regions"}`;
  for (const [index, roi] of state.diagnosticRois.entries()) {
    const row = document.createElement("div");
    row.className = "roi-list-row";

    const swatch = document.createElement("span");
    swatch.className = `roi-swatch ${roiClass(roi.type)}`;

    const text = document.createElement("div");
    text.className = "roi-list-text";
    const title = document.createElement("strong");
    title.textContent = `ROI ${index + 1} · ${roiTypeLabels[roi.type] || roiTypeLabels.other}`;
    const detail = document.createElement("span");
    const coords = `x ${Math.round(roi.x)}, y ${Math.round(roi.y)}, ${Math.round(roi.width)} × ${Math.round(roi.height)} px`;
    detail.textContent = roi.note ? `${roi.note} · ${coords}` : coords;
    text.append(title, detail);

    const focus = document.createElement("button");
    focus.type = "button";
    focus.className = "roi-icon-button";
    focus.title = "Focus ROI";
    focus.setAttribute("aria-label", `Focus ROI ${index + 1}`);
    focus.textContent = "◎";
    focus.onclick = () => focusRoi(roi);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "roi-icon-button danger";
    remove.title = "Delete ROI";
    remove.setAttribute("aria-label", `Delete ROI ${index + 1}`);
    remove.textContent = "×";
    remove.disabled = state.reportLocked;
    remove.onclick = () => removeRoi(roi.id);

    row.append(swatch, text, focus, remove);
    list.appendChild(row);
  }
}

function setRoiDrawing(enabled) {
  state.roiDrawing = Boolean(
    enabled && !state.reportLocked && state.viewer?.world.getItemCount()
  );
  state.roiStart = null;
  if (state.roiDraftElement) {
    state.viewer.removeOverlay(state.roiDraftElement);
    state.roiDraftElement = null;
  }
  state.viewer?.setMouseNavEnabled(!state.roiDrawing);
  const button = $("drawRoi");
  if (button) {
    button.classList.toggle("active", state.roiDrawing);
    button.textContent = state.roiDrawing ? "Drawing…" : "Draw ROI";
  }
}

function viewportRectFromPoints(a, b) {
  return new OpenSeadragon.Rect(
    Math.min(a.x, b.x),
    Math.min(a.y, b.y),
    Math.abs(a.x - b.x),
    Math.abs(a.y - b.y),
  );
}

function setupRoiTracker() {
  state.roiTracker = new OpenSeadragon.MouseTracker({
    element: state.viewer.canvas,
    pressHandler: event => {
      if (!state.roiDrawing || !state.viewer.world.getItemCount()) return;
      event.preventDefaultAction = true;
      state.roiStart = state.viewer.viewport.pointFromPixel(event.position, true);
      state.roiDraftElement = makeRoiOverlay({type: getVal("roiType")}, true);
      state.viewer.addOverlay({
        element: state.roiDraftElement,
        location: new OpenSeadragon.Rect(state.roiStart.x, state.roiStart.y, 0, 0),
        checkResize: false,
      });
    },
    dragHandler: event => {
      if (!state.roiDrawing || !state.roiStart || !state.roiDraftElement) return;
      event.preventDefaultAction = true;
      const point = state.viewer.viewport.pointFromPixel(event.position, true);
      state.viewer.updateOverlay(
        state.roiDraftElement,
        viewportRectFromPoints(state.roiStart, point),
      );
    },
    releaseHandler: event => {
      if (!state.roiDrawing || !state.roiStart || !state.roiDraftElement) return;
      event.preventDefaultAction = true;
      const end = state.viewer.viewport.pointFromPixel(event.position, true);
      const viewportRect = viewportRectFromPoints(state.roiStart, end);
      const item = state.viewer.world.getItemAt(0);
      const imageRect = item.viewportToImageRectangle(viewportRect);
      const dims = state.meta.dimensions;
      const x = Math.max(0, Math.min(dims[0], imageRect.x));
      const y = Math.max(0, Math.min(dims[1], imageRect.y));
      const width = Math.max(0, Math.min(dims[0] - x, imageRect.width));
      const height = Math.max(0, Math.min(dims[1] - y, imageRect.height));

      state.viewer.removeOverlay(state.roiDraftElement);
      state.roiDraftElement = null;
      state.roiStart = null;
      if (width >= 64 && height >= 64) {
        const type = getVal("roiType") || "diagnostic";
        state.diagnosticRois.push({
          id: `roi_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`,
          type,
          note: getVal("roiNote"),
          x: Math.round(x),
          y: Math.round(y),
          width: Math.round(width),
          height: Math.round(height),
          normalized: {
            x: x / dims[0],
            y: y / dims[1],
            width: width / dims[0],
            height: height / dims[1],
          },
          coordinate_space: "wsi_level_0_pixels",
          created_at: new Date().toISOString(),
        });
        $("roiNote").value = "";
        renderRoiList();
        renderRoiOverlays();
        markDirty();
      }
      setRoiDrawing(false);
    },
  });
  state.roiTracker.setTracking(true);
}

function updateViewerScale() {
  if (!state.viewer?.viewport || !state.viewer.world.getItemCount()) return;
  const home = state.viewer.viewport.getHomeZoom();
  const current = state.viewer.viewport.getZoom(true);
  const relative = home > 0 ? current / home : 1;
  $("viewerScale").textContent = `${relative.toFixed(relative < 10 ? 1 : 0)}× relative zoom`;
}

function initViewer() {
  state.viewer = OpenSeadragon({
    id: "wsiViewer",
    showNavigationControl: false,
    showNavigator: true,
    navigatorPosition: "BOTTOM_RIGHT",
    navigatorAutoFade: false,
    navigatorSizeRatio: 0.18,
    preserveViewport: false,
    visibilityRatio: 0.55,
    constrainDuringPan: true,
    minZoomImageRatio: 0.7,
    maxZoomPixelRatio: 4,
    animationTime: 0.35,
    blendTime: 0.08,
    imageLoaderLimit: 6,
    timeout: 120000,
    gestureSettingsMouse: {
      clickToZoom: false,
      dblClickToZoom: true,
      scrollToZoom: true,
      dragToPan: true,
    },
  });
  state.viewer.addHandler("open", () => {
    state.viewer.viewport.goHome(true);
    renderRoiOverlays();
    updateViewerScale();
  });
  state.viewer.addHandler("animation", updateViewerScale);
  state.viewer.addHandler("zoom", updateViewerScale);
  state.viewer.addHandler("open-failed", event => {
    console.error(event);
    $("viewerScale").textContent = "Slide failed to load";
  });
  setupRoiTracker();
}

async function save(status) {
  if (!state.readerProfile?.reader_id) return alert("Start a reader session first.");
  if (!state.activeCase) return alert("Select a case first.");
  if (state.reportLocked) return alert("This report has already been submitted and is locked.");
  clearTimeout(state.autosaveTimer);
  clearTimeout(state.autosaveRetryTimer);
  await waitForAutosaveIdle();
  clearTimeout(state.autosaveTimer);
  const localRecord = writeLocalDraft();
  try {
    const payload = collectAnnotation(status);
    payload.confirm_edit_submitted = state.editingSubmitted;
    const ann = await api(readerApiPath(`/api/annotation/${state.activeCase}`), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const submitted = ann.status === "submitted";
    clearLocalDraft(state.activeCase, state.readerProfile.reader_id);
    state.meta.annotation = ann;
    state.dirty = false;
    state.editingSubmitted = false;
    setReportLocked(submitted);
    setSyncStatus(
      submitted ? "Submitted and locked" : `Draft synced ${new Date().toLocaleTimeString()}`,
      "synced",
      "Synced",
    );
    const row = state.cases.find(c => c.case_id === state.activeCase);
    if (row) {
      row.annotation_status = ann.status;
      row.updated_at = ann.updated_at;
    }
    renderCases();
  } catch (err) {
    if (!localRecord) writeLocalDraft();
    setSyncStatus("Save failed; draft remains safe on this device", "offline", "Offline; draft safe");
    scheduleAutosaveRetry();
    alert(`Could not sync report: ${err.message}\nYour draft is saved on this device and will retry automatically.`);
  }
}

function profileFromSessionForm() {
  const readerId = getVal("sessionReaderId") || crypto.randomUUID();
  if (!/^[A-Za-z0-9_.-]{1,64}$/.test(readerId)) {
    throw new Error("Reader ID must use 1-64 letters, numbers, dots, underscores, or hyphens.");
  }
  return {reader_id: readerId};
}

function fillSessionForm(profile) {
  setVal("sessionReaderId", profile?.reader_id);
}

async function activateReader(profile, persist = true) {
  if (!/^[A-Za-z0-9_.-]{1,64}$/.test(profile?.reader_id || "")) {
    throw new Error("Invalid saved Reader ID.");
  }
  if (state.dirty) writeLocalDraft();
  state.readerProfile = profile;
  fillSessionForm(profile);
  if (persist) localStorage.setItem(readerStorageKey, JSON.stringify(profile));
  $("activeReaderStatus").textContent = `Active local session: ${profile.reader_id}`;
  state.activeCase = null;
  state.meta = null;
  state.cases = [];
  renderCases();
  fillAnnotation(null);
  state.editingSubmitted = false;
  setReportLocked(false);
  await loadCases();
}

async function activateReaderFromForm() {
  try {
    await activateReader(profileFromSessionForm());
  } catch (err) {
    alert(err.message);
  }
}

async function restoreReader() {
  const raw = localStorage.getItem(readerStorageKey);
  if (!raw) {
    $("statusLine").textContent = "Start a local session to begin.";
    return;
  }
  try {
    const profile = JSON.parse(raw);
    fillSessionForm(profile);
    await activateReader(profile, false);
  } catch (err) {
    localStorage.removeItem(readerStorageKey);
    $("statusLine").textContent = `Reader session error: ${err.message}`;
  }
}

function setupEvents() {
  initViewer();
  $("caseSearch").oninput = renderCases;
  document.querySelectorAll(".chip").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll(".chip").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.filter = btn.dataset.filter;
      renderCases();
    };
  });
  $("zoomIn").onclick = () => {
    if (!state.viewer.world.getItemCount()) return;
    state.viewer.viewport.zoomBy(1.5);
    state.viewer.viewport.applyConstraints();
  };
  $("zoomOut").onclick = () => {
    if (!state.viewer.world.getItemCount()) return;
    state.viewer.viewport.zoomBy(1 / 1.5);
    state.viewer.viewport.applyConstraints();
  };
  $("resetView").onclick = () => {
    if (!state.viewer.world.getItemCount()) return;
    state.viewer.viewport.goHome();
  };
  $("openViewerWindow").onclick = openViewerWindow;
  $("toggleViewerFullscreen").onclick = toggleViewerFullscreen;
  $("drawRoi").onclick = () => setRoiDrawing(!state.roiDrawing);
  $("applyReader").onclick = activateReaderFromForm;
  editableReportIds().forEach(id => {
    $(id).addEventListener("input", markDirty);
    $(id).addEventListener("change", markDirty);
  });
  $("saveDraft").onclick = () => save("draft");
  $("submitReport").onclick = () => {
    if (state.reportLocked) return unlockSubmittedForEditing();
    if (confirm("Submit and mark this report as locked/submitted?")) save("submitted");
  };
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
}

viewerWindowChannel?.addEventListener("message", (event) => {
  if (event.data?.type === "viewer-ready") publishViewerWindowCase();
});
document.addEventListener("fullscreenchange", updateViewerFullscreenButton);

setupEvents();
restoreReader().catch(err => {
  console.error(err);
  $("statusLine").textContent = `Error: ${err.message}`;
});
