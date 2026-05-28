const serverEl = document.getElementById("server");
const tokenEl = document.getElementById("token");
const autoplayEl = document.getElementById("autoplay");
const saveEl = document.getElementById("save");
const statusEl = document.getElementById("status");

async function load() {
  const { patmirServer, patmirToken, patmirAutoPlayEnabled } = await chrome.storage.local.get([
    "patmirServer", "patmirToken", "patmirAutoPlayEnabled",
  ]);
  serverEl.value = patmirServer || "http://127.0.0.1:8765";
  tokenEl.value = patmirToken || "";
  // Default ON.
  autoplayEl.checked = patmirAutoPlayEnabled !== false;
}

async function save() {
  const server = serverEl.value.trim() || "http://127.0.0.1:8765";
  const token = tokenEl.value.trim();
  if (!token) {
    statusEl.textContent = "Token is required.";
    statusEl.className = "err";
    return;
  }
  await chrome.storage.local.set({
    patmirServer: server,
    patmirToken: token,
    patmirAutoPlayEnabled: autoplayEl.checked,
  });
  statusEl.textContent = "Saved.";
  statusEl.className = "ok";
  setTimeout(() => { statusEl.textContent = ""; }, 2000);
}

saveEl.addEventListener("click", save);
document.addEventListener("DOMContentLoaded", load);
