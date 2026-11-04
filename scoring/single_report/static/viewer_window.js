const channelName = "pathxai-wsi-viewer-v1";
const storageKey = "pathxai-wsi-viewer-case-v1";
const channel = "BroadcastChannel" in window ? new BroadcastChannel(channelName) : null;
let activeCaseId = null;

const viewer = OpenSeadragon({
  id: "standaloneViewer",
  showNavigationControl: false,
  showNavigator: true,
  navigatorPosition: "BOTTOM_RIGHT",
  navigatorAutoFade: false,
  navigatorSizeRatio: 0.18,
  preserveViewport: false,
  visibilityRatio: 0.55,
  maxZoomPixelRatio: 2.5,
  timeout: 120000,
  gestureSettingsMouse: {
    clickToZoom: false,
    dblClickToZoom: true,
    scrollToZoom: true,
    dragToPan: true,
  },
});

function openCase(message) {
  if (!message?.case_id) return;
  document.getElementById("viewerWindowSite").textContent = `Known site: ${message.site || "not provided"}`;
  if (message.case_id === activeCaseId && viewer.world.getItemCount()) return;
  activeCaseId = message.case_id;
  document.getElementById("viewerWindowStatus").textContent = "Loading slide...";
  document.getElementById("viewerWindowPlaceholder")?.remove();
  viewer.open(`/api/dzi/${encodeURIComponent(activeCaseId)}.dzi`);
}

function readCurrentCase() {
  try { return JSON.parse(localStorage.getItem(storageKey)); }
  catch (_) { return null; }
}

viewer.addHandler("open", () => {
  viewer.viewport.goHome(true);
  document.getElementById("viewerWindowStatus").textContent = "Ready";
});
viewer.addHandler("open-failed", () => {
  document.getElementById("viewerWindowStatus").textContent = "Slide failed to load";
});

channel?.addEventListener("message", (event) => {
  if (event.data?.type === "open-case") openCase(event.data);
});
window.addEventListener("storage", (event) => {
  if (event.key === storageKey && event.newValue) {
    try { openCase(JSON.parse(event.newValue)); } catch (_) { /* Ignore malformed browser state. */ }
  }
});

document.getElementById("zoomIn").onclick = () => {
  if (!viewer.world.getItemCount()) return;
  viewer.viewport.zoomBy(1.5);
  viewer.viewport.applyConstraints();
};
document.getElementById("zoomOut").onclick = () => {
  if (!viewer.world.getItemCount()) return;
  viewer.viewport.zoomBy(1 / 1.5);
  viewer.viewport.applyConstraints();
};
document.getElementById("resetView").onclick = () => viewer.viewport.goHome();
document.getElementById("closeWindow").onclick = () => window.close();
document.getElementById("toggleFullscreen").onclick = async () => {
  if (document.fullscreenElement) await document.exitFullscreen();
  else await document.body.requestFullscreen();
};
document.addEventListener("fullscreenchange", () => {
  document.getElementById("toggleFullscreen").textContent = document.fullscreenElement ? "Exit full screen" : "Full screen";
  setTimeout(() => viewer.viewport.resize(), 50);
});

openCase(readCurrentCase());
channel?.postMessage({type: "viewer-ready"});
