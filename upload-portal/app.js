const CATEGORIES = [
  {
    slug: "locomotion",
    label: "Locomotion",
    icon: "🚶",
    desc: "Walk (front/side/3-4th angle), run, idle/breathing loop, head turn, full body turn.",
    target: 50,
  },
  {
    slug: "gestures",
    label: "Gestures",
    icon: "👋",
    desc: "Wave, point, reach, object pick-up, opening a door, checking a phone.",
    target: 30,
  },
  {
    slug: "expressions",
    label: "Facial Expressions",
    icon: "🙂",
    desc: "Blink, smile, frown/sad, surprise, talking/mouth-flap (silent), angry.",
    target: 40,
  },
  {
    slug: "secondary-motion",
    label: "Secondary Motion",
    icon: "🍃",
    desc: "Hair sway, cloth/dupatta movement, falling leaf, water ripple, curtain.",
    target: 20,
  },
];

const AUTO_TAB = { slug: "auto", label: "Smart Upload", icon: "⚡" };
const OVERALL_GOAL = 400;

// Instant filename-based guess, shown immediately while the real AI vision
// classification (see classifyWithVision below) is still running -- and used
// as the fallback if that call fails or the key isn't configured.
const CATEGORY_KEYWORDS = {
  locomotion: ["walk", "run", "jog", "sprint", "idle", "breath", "turn", "stride", "step", "locomotion", "gait"],
  gestures: ["wave", "point", "reach", "pickup", "pick-up", "pick_up", "door", "phone", "gesture", "grab", "hold", "open"],
  expressions: ["blink", "smile", "frown", "sad", "surprise", "surprised", "talk", "mouth", "flap", "angry", "expression", "cry", "laugh", "shock", "face"],
  "secondary-motion": ["hair", "sway", "cloth", "dupatta", "leaf", "water", "ripple", "curtain", "wind", "fabric", "scarf", "skirt", "secondary"],
};

const VIDEO_EXT_RE = /\.(mp4|mov|webm|mkv|avi|m4v|3gp|3gpp|wmv|flv|mts|m2ts)$/i;

const CLASSIFY_CONCURRENCY = 2;

const counts = {};
const staging = new Map();
let stagingIdSeq = 0;
let activeClassifyCount = 0;
const classifyQueue = [];

function isVideoFile(file) {
  if (file.type) return file.type.startsWith("video/");
  return VIDEO_EXT_RE.test(file.name);
}

function guessCategory(filename) {
  const lower = filename.toLowerCase();
  for (const cat of CATEGORIES) {
    const keywords = CATEGORY_KEYWORDS[cat.slug] || [];
    if (keywords.some((kw) => lower.includes(kw))) return cat.slug;
  }
  return null;
}

// Grabs one downscaled JPEG frame from the middle of the clip and returns it
// as base64 (no "data:" prefix), small enough to send to a vision model.
function extractFrameBase64(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    const url = URL.createObjectURL(file);
    video.src = url;

    const cleanup = () => URL.revokeObjectURL(url);
    const fail = (err) => { cleanup(); reject(err); };

    video.addEventListener("loadedmetadata", () => {
      const mid = (video.duration || 0) / 2;
      video.currentTime = isFinite(mid) && mid > 0 ? mid : 0;
    });

    video.addEventListener("seeked", () => {
      try {
        const w = video.videoWidth || 384;
        const h = video.videoHeight || 216;
        const scale = Math.min(1, 384 / w);
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(w * scale));
        canvas.height = Math.max(1, Math.round(h * scale));
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
        cleanup();
        resolve(dataUrl.split(",")[1]);
      } catch (err) {
        fail(err);
      }
    });

    video.addEventListener("error", () => fail(new Error("Could not read video frame")));
    video.load();
  });
}

async function classifyWithVision(file) {
  const imageBase64 = await extractFrameBase64(file);
  const res = await fetch("/api/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageBase64 }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Classify failed");
  return data.category || null;
}

function scheduleClassify(id, item) {
  classifyQueue.push({ id, item });
  pumpClassifyQueue();
}

function pumpClassifyQueue() {
  while (activeClassifyCount < CLASSIFY_CONCURRENCY && classifyQueue.length > 0) {
    const { id, item } = classifyQueue.shift();
    activeClassifyCount++;
    runClassify(id, item).finally(() => {
      activeClassifyCount--;
      pumpClassifyQueue();
    });
  }
}

async function runClassify(id, item) {
  if (!staging.has(id) || item.status !== "pending") return;
  item.analyzing = true;
  renderStagingList();

  try {
    const category = await classifyWithVision(item.file);
    if (staging.has(id) && item.status === "pending" && !item.manualOverride) {
      if (category) {
        item.slug = category;
        item.detectedBy = "ai";
      } else {
        // Model looked at the frame and genuinely wasn't confident -- keep
        // whatever filename guess exists, but say AI was consulted.
        item.detectedBy = "ai-unsure";
      }
    }
  } catch {
    // Frame extraction or the API call itself failed (unsupported codec,
    // network, missing key). This must be visible, not silent -- a silent
    // fallback here is exactly the "galat ho gaya, pata bhi nahi chala"
    // failure mode that matters for training data quality.
    if (staging.has(id) && item.status === "pending" && !item.manualOverride) {
      item.aiFailed = true;
    }
  } finally {
    if (staging.has(id)) item.analyzing = false;
    renderStagingList();
  }
}

function fmtBytes(bytes) {
  if (!bytes) return "";
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`;
}

function fmtDuration(seconds) {
  if (seconds === undefined || seconds === null || isNaN(seconds)) return "";
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function timeAgo(iso) {
  if (!iso) return "";
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.floor(months / 12)}y ago`;
}

function buildTabs() {
  const tabsEl = document.getElementById("tabs");
  const panelsEl = document.getElementById("panels");
  const allTabs = [AUTO_TAB, ...CATEGORIES];

  allTabs.forEach((cat, i) => {
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (i === 0 ? " active" : "");
    btn.innerHTML = `<span class="tab-icon">${cat.icon}</span> ${cat.label}`;
    btn.dataset.slug = cat.slug;
    btn.onclick = () => activateTab(cat.slug);
    tabsEl.appendChild(btn);

    const panel = document.createElement("section");
    panel.className = "panel" + (i === 0 ? " active" : "");
    panel.id = `panel-${cat.slug}`;

    if (cat.slug === "auto") {
      panel.innerHTML = buildAutoPanelHTML();
      panelsEl.appendChild(panel);
      wireAutoDropzone();
    } else {
      panel.innerHTML = `
        <div class="panel-header">
          <h2>${cat.icon} ${cat.label}</h2>
          <span class="count-badge" id="badge-${cat.slug}">0 / ${cat.target}</span>
        </div>
        <p class="panel-desc">${cat.desc}</p>
        <div class="dropzone" id="dropzone-${cat.slug}">
          <div class="dropzone-icon">🎬</div>
          <div class="dropzone-text"><strong>Videos drag & drop karo</strong> ya click karke chuno</div>
          <div class="dropzone-hint">.mp4 .mov .webm .mkv &mdash; sirf video, images/photos allowed nahi hain</div>
          <input type="file" id="input-${cat.slug}" accept="video/*" multiple />
        </div>
        <div class="upload-queue" id="queue-${cat.slug}"></div>
        <div class="grid" id="grid-${cat.slug}"></div>
      `;
      panelsEl.appendChild(panel);
      wireDropzone(cat.slug);
    }
  });
}

function activateTab(slug) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.slug === slug));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${slug}`));
}

// --- Smart Upload (auto-detect) tab -----------------------------------

function buildAutoPanelHTML() {
  return `
    <div class="panel-header">
      <h2>⚡ Smart Upload</h2>
      <span class="count-badge">Auto category-detect</span>
    </div>
    <p class="panel-desc">
      Sirf videos yahan daal do — ek AI vision model video ka frame dekh ke category khud
      guess kar lega (🤖 AI-detected), filename bhi turant ek hint deta hai (🔍) jab tak AI check kar raha ho.
      Guess galat lage toh dropdown se badal do, phir "Upload All" dabao.
    </p>
    <div class="dropzone" id="dropzone-auto">
      <div class="dropzone-icon">🎬</div>
      <div class="dropzone-text"><strong>Videos yahan daalo</strong> ya click karke chuno</div>
      <div class="dropzone-hint">.mp4 .mov .webm .mkv &mdash; sirf video, images/photos allowed nahi hain</div>
      <input type="file" id="input-auto" accept="video/*" multiple />
    </div>
    <div id="staging-list" class="staging-list"></div>
    <div id="staging-actions" class="staging-actions" style="display:none;">
      <span id="staging-hint" class="staging-hint"></span>
      <div class="staging-buttons">
        <button id="staging-clear" class="btn-secondary" type="button">Clear</button>
        <button id="staging-upload-all" class="btn-primary" type="button">Upload All</button>
      </div>
    </div>
  `;
}

function wireAutoDropzone() {
  const zone = document.getElementById("dropzone-auto");
  const input = document.getElementById("input-auto");

  zone.onclick = () => input.click();
  input.onchange = () => {
    addToStaging(input.files);
    input.value = "";
  };

  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
    })
  );
  zone.addEventListener("drop", (e) => addToStaging(e.dataTransfer.files));

  document.getElementById("staging-clear").onclick = () => {
    staging.clear();
    renderStagingList();
  };
  document.getElementById("staging-upload-all").onclick = uploadAllStaging;
}

function addToStaging(fileList) {
  const files = Array.from(fileList);
  for (const file of files) {
    const id = ++stagingIdSeq;
    if (!isVideoFile(file)) {
      staging.set(id, { file, slug: null, status: "invalid" });
      continue;
    }
    const item = { file, slug: guessCategory(file.name), status: "pending", progress: 0, detectedBy: null, analyzing: false };
    staging.set(id, item);
    scheduleClassify(id, item);
  }
  renderStagingList();
}

function statusLabel(item) {
  if (item.status === "pending") {
    if (item.analyzing) return "🔎 AI dekh raha hai…";
    if (item.detectedBy === "ai") return "🤖 AI-detected";
    if (item.detectedBy === "manual") return "✋ manually set";
    if (item.aiFailed) return item.slug ? "⚠️ AI fail hua — filename guess" : "⚠️ AI fail hua — pick karo";
    if (item.detectedBy === "ai-unsure") return item.slug ? "🔍 filename guess (AI unsure)" : "❓ AI unsure — pick karo";
    return item.slug ? "🔍 filename guess" : "❓ pick karo";
  }
  if (item.status === "uploading") return `Uploading… ${item.progress}%`;
  if (item.status === "done") return "✅ Uploaded";
  if (item.status === "error") return `❌ ${item.errorMsg || "Failed"}`;
  return "";
}

function renderStagingList() {
  const listEl = document.getElementById("staging-list");
  const actionsEl = document.getElementById("staging-actions");
  const hintEl = document.getElementById("staging-hint");
  const uploadBtn = document.getElementById("staging-upload-all");

  if (staging.size === 0) {
    listEl.innerHTML = "";
    actionsEl.style.display = "none";
    return;
  }

  actionsEl.style.display = "flex";
  listEl.innerHTML = "";

  let pendingCount = 0;
  let needsCategory = 0;

  staging.forEach((item, id) => {
    const row = document.createElement("div");
    row.className = "staging-row" +
      (item.status === "invalid" || item.status === "error" ? " error" : "") +
      (item.status === "done" ? " done" : "") +
      (item.status === "pending" && item.aiFailed ? " warn" : "");
    row.dataset.id = id;

    if (item.status === "invalid") {
      row.innerHTML = `
        <div class="row-top">
          <span class="row-name" title="${item.file.name}">${item.file.name}</span>
          <span class="row-status">❌ Image/doc hai, video nahi</span>
        </div>
        <div class="staging-row-bottom">
          <span></span>
          <button class="row-remove" type="button" title="Remove">✕</button>
        </div>
      `;
    } else {
      if (item.status === "pending") {
        pendingCount += 1;
        if (!item.slug) needsCategory += 1;
      }

      const options = CATEGORIES.map(
        (c) => `<option value="${c.slug}" ${item.slug === c.slug ? "selected" : ""}>${c.icon} ${c.label}</option>`
      ).join("");

      row.innerHTML = `
        <div class="row-top">
          <span class="row-name" title="${item.file.name}">${item.file.name}</span>
          <span class="row-status">${statusLabel(item)}</span>
        </div>
        <div class="staging-row-bottom">
          <select class="staging-select" ${item.status !== "pending" ? "disabled" : ""} ${!item.slug ? 'data-empty="1"' : ""}>
            <option value="" ${!item.slug ? "selected" : ""} disabled>❓ Category chuno</option>
            ${options}
          </select>
          <button class="row-remove" type="button" title="Remove" ${item.status === "uploading" ? "disabled" : ""}>✕</button>
        </div>
        <div class="row-bar"><div class="row-bar-fill" style="width:${item.progress || 0}%"></div></div>
      `;

      row.querySelector(".staging-select").onchange = (e) => {
        item.slug = e.target.value || null;
        item.manualOverride = true;
        item.detectedBy = "manual";
        renderStagingList();
      };
      row.querySelector(".row-remove").onclick = () => {
        staging.delete(id);
        renderStagingList();
      };
    }

    listEl.appendChild(row);
  });

  if (needsCategory > 0) {
    hintEl.textContent = `⚠️ ${needsCategory} file(s) ko category chunni baaki hai`;
    uploadBtn.disabled = true;
  } else if (pendingCount === 0) {
    hintEl.textContent = "Sab ho gaya ✅";
    uploadBtn.disabled = true;
  } else {
    hintEl.textContent = `${pendingCount} video ready to upload`;
    uploadBtn.disabled = false;
  }
}

function updateStagingRowProgress(id, pct) {
  const row = document.querySelector(`#staging-list [data-id="${id}"]`);
  if (!row) return;
  const bar = row.querySelector(".row-bar-fill");
  const status = row.querySelector(".row-status");
  if (bar) bar.style.width = `${pct}%`;
  if (status) status.textContent = `Uploading… ${pct}%`;
}

async function uploadAllStaging() {
  const touchedSlugs = new Set();
  const entries = Array.from(staging.entries()).filter(([, item]) => item.status === "pending" && item.slug);

  for (const [id, item] of entries) {
    item.status = "uploading";
    item.progress = 0;
    renderStagingList();
    try {
      await uploadOne(item.slug, item.file, (pct) => {
        item.progress = pct;
        updateStagingRowProgress(id, pct);
      });
      item.status = "done";
      touchedSlugs.add(item.slug);
    } catch (err) {
      item.status = "error";
      item.errorMsg = err.message;
    }
    renderStagingList();
  }

  for (const slug of touchedSlugs) await loadCategory(slug);
}

// --- Per-category tabs (browse / manual upload) ------------------------

function wireDropzone(slug) {
  const zone = document.getElementById(`dropzone-${slug}`);
  const input = document.getElementById(`input-${slug}`);

  zone.onclick = () => input.click();
  input.onchange = () => {
    handleFiles(slug, input.files);
    input.value = "";
  };

  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
    })
  );
  zone.addEventListener("drop", (e) => handleFiles(slug, e.dataTransfer.files));
}

function makeQueueRow(queueEl, file) {
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `
    <div class="row-top">
      <span class="row-name" title="${file.name}">${file.name}</span>
      <span class="row-status">Queued…</span>
    </div>
    <div class="row-bar"><div class="row-bar-fill"></div></div>
  `;
  queueEl.appendChild(row);
  return {
    row,
    statusEl: row.querySelector(".row-status"),
    barFill: row.querySelector(".row-bar-fill"),
  };
}

async function handleFiles(slug, fileList) {
  const files = Array.from(fileList);
  const queueEl = document.getElementById(`queue-${slug}`);
  queueEl.innerHTML = "";

  let uploadedAny = false;

  for (const file of files) {
    const { row, statusEl, barFill } = makeQueueRow(queueEl, file);

    if (!isVideoFile(file)) {
      row.classList.add("error");
      statusEl.textContent = "❌ Ye image/document hai — sirf video files allowed hain";
      barFill.style.width = "100%";
      continue;
    }

    statusEl.textContent = "0%";
    try {
      await uploadOne(slug, file, (pct) => {
        statusEl.textContent = `Uploading… ${pct}%`;
        barFill.style.width = `${pct}%`;
      });
      row.classList.add("done");
      statusEl.textContent = "✅ Uploaded";
      barFill.style.width = "100%";
      uploadedAny = true;
    } catch (err) {
      row.classList.add("error");
      statusEl.textContent = `❌ Failed: ${err.message}`;
    }
  }

  if (uploadedAny) await loadCategory(slug);
}

function uploadOne(slug, file, onProgress) {
  return new Promise((resolve, reject) => {
    fetch("/api/sign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag: slug }),
    })
      .then(async (signRes) => {
        const signData = await signRes.json();
        if (!signRes.ok) throw new Error(signData.error || "Could not sign upload");

        const form = new FormData();
        form.append("file", file);
        form.append("api_key", signData.apiKey);
        form.append("timestamp", signData.timestamp);
        form.append("signature", signData.signature);
        form.append("folder", signData.folder);
        form.append("tags", signData.tags);

        const xhr = new XMLHttpRequest();
        xhr.open("POST", `https://api.cloudinary.com/v1_1/${signData.cloudName}/video/upload`);
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
        };
        xhr.onload = () => {
          let data = {};
          try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
          if (xhr.status >= 200 && xhr.status < 300) resolve(data);
          else reject(new Error(data.error?.message || "Cloudinary upload failed"));
        };
        xhr.onerror = () => reject(new Error("Network error during upload"));
        xhr.send(form);
      })
      .catch(reject);
  });
}

async function loadCategory(slug) {
  const grid = document.getElementById(`grid-${slug}`);
  const badge = document.getElementById(`badge-${slug}`);
  const cat = CATEGORIES.find((c) => c.slug === slug);

  const res = await fetch(`/api/list?tag=${encodeURIComponent(slug)}`);
  const data = await res.json();
  if (!res.ok) {
    grid.innerHTML = `<div class="empty-state">⚠️ Could not load clips: ${data.error || "unknown error"}</div>`;
    return;
  }

  const clips = data.clips || [];
  counts[slug] = clips.length;
  badge.textContent = `${clips.length} / ${cat.target}`;

  if (clips.length === 0) {
    grid.innerHTML = `<div class="empty-state">🎥 Abhi tak koi video nahi hai — pehla upload tum karo!</div>`;
  } else {
    grid.innerHTML = "";
    clips
      .slice()
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
      .forEach((clip) => grid.appendChild(renderCard(slug, clip)));
  }

  updateOverallProgress();
}

function renderCard(slug, clip) {
  const card = document.createElement("div");
  card.className = "card";

  const name = clip.publicId.split("/").pop();
  const duration = fmtDuration(clip.duration);
  const ago = timeAgo(clip.createdAt);

  card.innerHTML = `
    <div class="thumb-wrap">
      <video class="thumb-video" src="${clip.url}#t=0.1" poster="${clip.thumbnailUrl}" muted loop preload="metadata" playsinline></video>
      <div class="play-overlay">▶</div>
      ${duration ? `<span class="duration-badge">${duration}</span>` : ""}
    </div>
    <div class="meta">
      <div class="meta-text">
        <span class="clip-name" title="${name}">${name}</span>
        <span class="clip-sub">${fmtBytes(clip.bytes)}${ago ? " • " + ago : ""}</span>
      </div>
      <button class="delete-btn" title="Delete this clip">🗑</button>
    </div>
  `;

  const video = card.querySelector("video");
  const thumbWrap = card.querySelector(".thumb-wrap");

  thumbWrap.addEventListener("mouseenter", () => {
    video.currentTime = 0;
    video.play().catch(() => {});
  });
  thumbWrap.addEventListener("mouseleave", () => {
    video.pause();
    video.currentTime = 0;
    video.controls = false;
    video.muted = true;
  });
  thumbWrap.addEventListener("click", () => {
    video.controls = true;
    video.muted = false;
    video.play().catch(() => {});
  });

  card.querySelector(".delete-btn").onclick = async (e) => {
    e.stopPropagation();
    if (!confirm(`Delete "${name}"? This can't be undone.`)) return;
    const res = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ publicId: clip.publicId }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(`Delete failed: ${data.error || "unknown error"}`);
      return;
    }
    await loadCategory(slug);
  };

  return card;
}

function updateOverallProgress() {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const el = document.getElementById("overall-progress");
  const pct = Math.min(100, (total / OVERALL_GOAL) * 100);
  el.innerHTML = `
    <div class="bar"><div class="bar-fill" style="width:${pct}%"></div></div>
    <p style="margin:6px 0 0;color:var(--muted);font-size:13px;">${total} / ${OVERALL_GOAL} clips uploaded &mdash; koi upload limit nahi hai, ye sirf progress count hai</p>
  `;
}

function init() {
  buildTabs();
  CATEGORIES.forEach((cat) => loadCategory(cat.slug));
}

init();
