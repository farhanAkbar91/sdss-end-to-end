const FEATURE_COLS = [
  { id: 'alpha', label: 'Right Ascension (alpha)', value: 180.0 },
  { id: 'delta', label: 'Declination (delta)', value: 0.0 },
  { id: 'u', label: 'u (Ultraviolet)', value: 23.0 },
  { id: 'g', label: 'g (Green)', value: 21.0 },
  { id: 'r', label: 'r (Red)', value: 20.0 },
  { id: 'i', label: 'i (Near Infrared)', value: 19.0 },
  { id: 'z', label: 'z (Infrared)', value: 18.0 },
  { id: 'cam_col', label: 'Camera Column (cam_col)', value: 3 },
  { id: 'redshift', label: 'Redshift', value: 0.0 },
  { id: 'plate', label: 'Plate ID', value: 4000 },
  { id: 'MJD', label: 'Modified Julian Date', value: 55000 }
];

const SAMPLES = {
  GALAXY: { alpha: 180.0, delta: 0.0, u: 23.0, g: 21.0, r: 20.0, i: 19.0, z: 18.0, cam_col: 3, redshift: 0.1, plate: 4000, MJD: 55000 },
  QSO: { alpha: 200.0, delta: 10.0, u: 19.0, g: 19.0, r: 19.0, i: 18.0, z: 18.0, cam_col: 4, redshift: 2.5, plate: 5000, MJD: 56000 },
  STAR: { alpha: 150.0, delta: 20.0, u: 25.0, g: 24.0, r: 23.0, i: 22.0, z: 21.0, cam_col: 2, redshift: 0.0001, plate: 3000, MJD: 54000 }
};

document.addEventListener("DOMContentLoaded", () => {
  // Navigation
  const menuItems = document.querySelectorAll(".menu-item");
  const pages = document.querySelectorAll(".page");

  menuItems.forEach(item => {
    item.addEventListener("click", () => {
      menuItems.forEach(m => m.classList.remove("active"));
      pages.forEach(p => p.classList.remove("active"));

      item.classList.add("active");
      const targetId = "page-" + item.dataset.page;
      const targetPage = document.getElementById(targetId);
      if (targetPage) targetPage.classList.add("active");
    });
  });

  // Render input form
  renderInputForm();
  
  // Stats
  const statTotal = document.getElementById("stat-total");
  const statFeatures = document.getElementById("stat-features");
  if (statTotal) statTotal.innerText = "100,000";
  if (statFeatures) statFeatures.innerText = FEATURE_COLS.length;
  
  // Event Listeners
  const btnPredict = document.getElementById("btn-predict");
  if (btnPredict) btnPredict.addEventListener("click", predictObject);
  
  const btnCompare = document.getElementById("btn-train-all");
  if (btnCompare) btnCompare.addEventListener("click", compareModels);
  

  
  document.querySelectorAll(".sample-btn").forEach(btn => {
    btn.addEventListener("click", (e) => loadSample(e.target.dataset.class));
  });

  // Hide static chart images since API doesn't provide them
  document.querySelectorAll('.chart-img').forEach(img => {
      img.style.display = 'none';
  });
  const loaders = document.querySelectorAll('.chart-loader');
  loaders.forEach(loader => loader.classList.add('hidden'));
});

function renderInputForm() {
  const formGrid = document.getElementById("input-form");
  if (!formGrid) return;
  
  formGrid.innerHTML = '';
  FEATURE_COLS.forEach(feat => {
    formGrid.innerHTML += `
      <div class="form-field">
        <label>${feat.label}</label>
        <input type="number" id="input-${feat.id}" value="${feat.value}" step="any">
      </div>
    `;
  });
}

function loadSample(className) {
  const sample = SAMPLES[className];
  if (!sample) return;
  for (const [key, value] of Object.entries(sample)) {
    const el = document.getElementById(`input-${key}`);
    if (el) el.value = value;
  }
}

async function predictObject() {
  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  const originalHtml = btn.innerHTML;
  btn.innerHTML = `<div class="spinner" style="width:16px;height:16px;margin-right:8px;border-width:2px;display:inline-block;vertical-align:middle;"></div> Memprediksi...`;
  
  const features = {};
  FEATURE_COLS.forEach(feat => {
    const el = document.getElementById(`input-${feat.id}`);
    features[feat.id] = el ? parseFloat(el.value) : feat.value;
  });
  
  const modelName = document.getElementById("pred-model").value;
  
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, model: modelName })
    });
    
    const data = await res.json();
    if (res.ok) {
      renderPredictionResult(data);
    } else {
      alert("Error: " + data.error);
    }
  } catch (err) {
    alert("Koneksi gagal. Pastikan backend Flask berjalan.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
  }
}

function renderPredictionResult(data) {
  const area = document.getElementById("predict-result-area");
  
  let emoji = "🌌";
  if (data.prediction === "STAR") emoji = "⭐";
  if (data.prediction === "GALAXY") emoji = "🌀";
  if (data.prediction === "QSO") emoji = "⚡";

  let probHtml = "";
  if (data.probabilities && Object.keys(data.probabilities).length > 0) {
    probHtml += `<table class="proba-table">
      <tr><th>Kelas</th><th>Probabilitas</th></tr>`;
    for (const [cls, prob] of Object.entries(data.probabilities)) {
      probHtml += `<tr><td>${cls}</td><td>${(prob * 100).toFixed(2)}%</td></tr>`;
    }
    probHtml += `</table>`;
  }
  
  area.innerHTML = `
    <div class="result-badge ${data.prediction}">
      ${emoji} ${data.prediction}
    </div>
    <p style="text-align:center; color:#8a8aad; margin-bottom: 16px;">Model: ${data.model}</p>
    ${probHtml}
  `;
}

async function compareModels() {
  const btn = document.getElementById("btn-train-all");
  const loader = document.getElementById("train-loader");
  const resultsArea = document.getElementById("compare-results");
  
  btn.style.display = "none";
  loader.classList.remove("hidden");
  resultsArea.classList.add("hidden");
  
  const features = {};
  FEATURE_COLS.forEach(feat => {
    const el = document.getElementById(`input-${feat.id}`);
    features[feat.id] = el ? parseFloat(el.value) : feat.value;
  });
  
  try {
    const res = await fetch('/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    
    const data = await res.json();
    if (res.ok) {
      renderCompareResult(data.results);
      resultsArea.classList.remove("hidden");
    } else {
      alert("Error: " + data.error);
    }
  } catch (err) {
    alert("Koneksi gagal.");
  } finally {
    btn.style.display = "flex";
    loader.classList.add("hidden");
  }
}

function renderCompareResult(results) {
  const wrap = document.getElementById("accuracy-table-wrap");
  if (!wrap) return;
  
  let tableHtml = `<table class="acc-table">
    <tr>
      <th>Model</th>
      <th>Prediksi</th>
      <th>Accuracy</th>
      <th>F1 Score</th>
    </tr>`;
    
  for (const [modelName, info] of Object.entries(results)) {
    const pred = info.prediction;
    const metrics = info.metrics || {};
    const acc = metrics.accuracy ? (metrics.accuracy * 100).toFixed(2) + "%" : "N/A";
    const f1 = metrics.f1_score ? (metrics.f1_score * 100).toFixed(2) + "%" : "N/A";
    
    let predStyle = "";
    if (pred === "STAR") predStyle = "color: var(--star);";
    if (pred === "GALAXY") predStyle = "color: var(--galaxy);";
    if (pred === "QSO") predStyle = "color: var(--qso);";
    
    tableHtml += `
      <tr>
        <td style="font-weight:bold; color:#fff;">${modelName}</td>
        <td style="font-weight:bold; ${predStyle}">${pred}</td>
        <td>${acc}</td>
        <td>${f1}</td>
      </tr>
    `;
  }
  tableHtml += `</table>`;
  wrap.innerHTML = tableHtml;
}

