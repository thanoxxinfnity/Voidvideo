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

const VIDEO_EXT_RE = /\.(mp4|mov|webm|mkv|avi|m4v|3gp|3gpp|wmv|flv|mts|m2ts)$/i;

const counts = {};

function isVideoFile(file) {
  if (file.type) return file.type.startsWith("video/");
  return VIDEO_EXT_RE.test(file.name);
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

  CATEGORIES.forEach((cat, i) => {
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (i === 0 ? " active" : "");
    btn.innerHTML = `<span class="tab-icon">${cat.icon}</span> ${cat.label}`;
    btn.dataset.slug = cat.slug;
    btn.onclick = () => activateTab(cat.slug);
    tabsEl.appendChild(btn);

    const panel = document.createElement("section");
    panel.className = "panel" + (i === 0 ? " active" : "");
    panel.id = `panel-${cat.slug}`;
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
  });
}

function activateTab(slug) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.slug === slug));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${slug}`));
}

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
  const pct = Math.min(100, (total / 300) * 100);
  el.innerHTML = `
    <div class="bar"><div class="bar-fill" style="width:${pct}%"></div></div>
    <p style="margin:6px 0 0;color:var(--muted);font-size:13px;">${total} total clips uploaded (goal: 150–300)</p>
  `;
}

function init() {
  buildTabs();
  CATEGORIES.forEach((cat) => loadCategory(cat.slug));
}

init();
