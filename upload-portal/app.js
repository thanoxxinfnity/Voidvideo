const CATEGORIES = [
  {
    slug: "locomotion",
    label: "Locomotion",
    desc: "Walk (front/side/3-4th angle), run, idle/breathing loop, head turn, full body turn.",
    target: 50,
  },
  {
    slug: "gestures",
    label: "Gestures",
    desc: "Wave, point, reach, object pick-up, opening a door, checking a phone.",
    target: 30,
  },
  {
    slug: "expressions",
    label: "Facial Expressions",
    desc: "Blink, smile, frown/sad, surprise, talking/mouth-flap (silent), angry.",
    target: 40,
  },
  {
    slug: "secondary-motion",
    label: "Secondary Motion",
    desc: "Hair sway, cloth/dupatta movement, falling leaf, water ripple, curtain.",
    target: 20,
  },
];

const counts = {};

function fmtBytes(bytes) {
  if (!bytes) return "";
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`;
}

function buildTabs() {
  const tabsEl = document.getElementById("tabs");
  const panelsEl = document.getElementById("panels");

  CATEGORIES.forEach((cat, i) => {
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (i === 0 ? " active" : "");
    btn.textContent = cat.label;
    btn.dataset.slug = cat.slug;
    btn.onclick = () => activateTab(cat.slug);
    tabsEl.appendChild(btn);

    const panel = document.createElement("section");
    panel.className = "panel" + (i === 0 ? " active" : "");
    panel.id = `panel-${cat.slug}`;
    panel.innerHTML = `
      <div class="panel-header">
        <h2>${cat.label}</h2>
        <span class="count-badge" id="badge-${cat.slug}">0 / ${cat.target}</span>
      </div>
      <p class="panel-desc">${cat.desc}</p>
      <div class="dropzone" id="dropzone-${cat.slug}">
        Drag & drop .mp4 clips here, or click to choose files
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
  input.onchange = () => handleFiles(slug, input.files);

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

async function handleFiles(slug, fileList) {
  const files = Array.from(fileList);
  const queueEl = document.getElementById(`queue-${slug}`);
  queueEl.innerHTML = "";

  for (const file of files) {
    const row = document.createElement("div");
    row.className = "row";
    row.textContent = `Uploading ${file.name}...`;
    queueEl.appendChild(row);

    try {
      await uploadOne(slug, file);
      row.classList.add("done");
      row.textContent = `${file.name} — uploaded`;
    } catch (err) {
      row.classList.add("error");
      row.textContent = `${file.name} — failed: ${err.message}`;
    }
  }

  await loadCategory(slug);
}

async function uploadOne(slug, file) {
  const signRes = await fetch("/api/sign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tag: slug }),
  });
  const signData = await signRes.json();
  if (!signRes.ok) throw new Error(signData.error || "Could not sign upload");

  const form = new FormData();
  form.append("file", file);
  form.append("api_key", signData.apiKey);
  form.append("timestamp", signData.timestamp);
  form.append("signature", signData.signature);
  form.append("folder", signData.folder);
  form.append("tags", signData.tags);

  const uploadRes = await fetch(`https://api.cloudinary.com/v1_1/${signData.cloudName}/video/upload`, {
    method: "POST",
    body: form,
  });
  const uploadData = await uploadRes.json();
  if (!uploadRes.ok) throw new Error(uploadData.error?.message || "Cloudinary upload failed");
  return uploadData;
}

async function loadCategory(slug) {
  const grid = document.getElementById(`grid-${slug}`);
  const badge = document.getElementById(`badge-${slug}`);
  const cat = CATEGORIES.find((c) => c.slug === slug);

  const res = await fetch(`/api/list?tag=${encodeURIComponent(slug)}`);
  const data = await res.json();
  if (!res.ok) {
    grid.innerHTML = `<div class="empty-state">Could not load clips: ${data.error || "unknown error"}</div>`;
    return;
  }

  const clips = data.clips || [];
  counts[slug] = clips.length;
  badge.textContent = `${clips.length} / ${cat.target}`;

  if (clips.length === 0) {
    grid.innerHTML = `<div class="empty-state">No clips uploaded yet in this category.</div>`;
  } else {
    grid.innerHTML = "";
    clips.forEach((clip) => grid.appendChild(renderCard(slug, clip)));
  }

  updateOverallProgress();
}

function renderCard(slug, clip) {
  const card = document.createElement("div");
  card.className = "card";

  const name = clip.publicId.split("/").pop();
  card.innerHTML = `
    <img src="${clip.thumbnailUrl}" alt="${name}" loading="lazy" />
    <div class="meta">
      <span title="${name}">${name} (${fmtBytes(clip.bytes)})</span>
      <button class="delete-btn">Delete</button>
    </div>
  `;

  card.querySelector(".delete-btn").onclick = async () => {
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
