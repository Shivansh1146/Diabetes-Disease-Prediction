/* ── script.js ── */

// ─── Theme Toggle ───────────────────────────────────
const THEME_KEY = 'dp_theme';
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_KEY, theme);
  const btn = document.getElementById('themeToggle');
  if (btn) btn.textContent = theme === 'dark' ? '☀️' : '🌙';
}
function initTheme() {
  const saved = localStorage.getItem(THEME_KEY) || 'dark';
  applyTheme(saved);
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme') || 'dark';
  applyTheme(cur === 'dark' ? 'light' : 'dark');
}
initTheme();

// ─── Sidebar Mobile Toggle ───────────────────────────
function toggleSidebar() {
  document.querySelector('.sidebar')?.classList.toggle('open');
}

// ─── Auto-dismiss flash messages ────────────────────
setTimeout(() => {
  document.querySelectorAll('.flash-alert').forEach(el => {
    el.style.transition = 'opacity 0.5s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 500);
  });
}, 4000);

// ─── Animate stat counters ──────────────────────────
function animateCounter(el) {
  const target = parseInt(el.getAttribute('data-target') || el.textContent, 10);
  if (isNaN(target)) return;
  let cur = 0;
  const step = Math.ceil(target / 40);
  const timer = setInterval(() => {
    cur = Math.min(cur + step, target);
    el.textContent = cur;
    if (cur >= target) clearInterval(timer);
  }, 30);
}
document.querySelectorAll('.stat-value[data-target]').forEach(animateCounter);

// ─── BMI Calculator ─────────────────────────────────
async function calcBMI() {
  const w = parseFloat(document.getElementById('bmiWeight')?.value);
  const h = parseFloat(document.getElementById('bmiHeight')?.value);
  if (!w || !h || w <= 0 || h <= 0) {
    showBMIResult('⚠️ Please enter valid weight and height.', 'warning');
    return;
  }
  try {
    const res = await fetch('/api/bmi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ weight: w, height: h })
    });
    const data = await res.json();
    const colors = { success:'#10b981', warning:'#f59e0b', danger:'#ef4444', info:'#3b82f6', secondary:'#94a3b8' };
    const color = colors[data.color] || '#4f46e5';
    showBMIResult(
      `BMI: <strong style="color:${color}">${data.bmi}</strong> — <em>${data.category}</em>`,
      data.color
    );
    const bmiInput = document.getElementById('bmi');
    if (bmiInput) bmiInput.value = data.bmi;
  } catch { showBMIResult('Error calculating BMI.', 'danger'); }
}
function showBMIResult(html, type) {
  const el = document.getElementById('bmiResult');
  if (!el) return;
  const bg = { success:'rgba(16,185,129,0.1)', warning:'rgba(245,158,11,0.1)', danger:'rgba(239,68,68,0.1)', info:'rgba(59,130,246,0.1)' };
  el.innerHTML = html;
  el.style.cssText = `display:block;padding:10px 14px;border-radius:8px;background:${bg[type]||'rgba(79,70,229,0.1)'};margin-top:10px;font-size:14px`;
}

// ─── Risk Progress Bar Animation ─────────────────────
function animateRiskBar() {
  const bar = document.getElementById('riskBar');
  if (!bar) return;
  const pct = parseFloat(bar.getAttribute('data-pct') || '0');
  const color = pct >= 75 ? '#ef4444' : pct >= 60 ? '#f97316' : pct >= 45 ? '#f59e0b' : pct >= 25 ? '#3b82f6' : '#10b981';
  bar.style.width = '0%';
  bar.style.background = color;
  setTimeout(() => { bar.style.width = pct + '%'; }, 200);
}
document.addEventListener('DOMContentLoaded', animateRiskBar);

// ─── Dashboard Charts ────────────────────────────────
function initDashboardCharts(labels, diabeticData, nonDiabeticData) {
  if (!window.Chart) return;
  const ctx = document.getElementById('trendChart');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Diabetic', data: diabeticData,
          borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.12)',
          tension: 0.4, fill: true, pointBackgroundColor: '#ef4444',
          pointRadius: 5, pointHoverRadius: 7,
        },
        {
          label: 'Non-Diabetic', data: nonDiabeticData,
          borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.12)',
          tension: 0.4, fill: true, pointBackgroundColor: '#10b981',
          pointRadius: 5, pointHoverRadius: 7,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#94a3b8', usePointStyle: true } } },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true }
      }
    }
  });
}

function initDistChart(diabetic, nonDiabetic) {
  const ctx = document.getElementById('distChart');
  if (!ctx || !window.Chart) return;
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Diabetic', 'Non-Diabetic'],
      datasets: [{
        data: [diabetic, nonDiabetic],
        backgroundColor: ['rgba(239,68,68,0.8)', 'rgba(16,185,129,0.8)'],
        borderColor: ['#ef4444', '#10b981'], borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '70%',
      plugins: { legend: { labels: { color: '#94a3b8', usePointStyle: true } } }
    }
  });
}

function initModelChart(metrics) {
  const ctx = document.getElementById('modelChart');
  if (!ctx || !window.Chart || !metrics) return;
  new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
      datasets: [{
        label: metrics.model_name || 'Best Model',
        data: [
          metrics.accuracy, metrics.precision, metrics.recall,
          metrics.f1_score, (metrics.auc_roc || 0.8) * 100
        ],
        backgroundColor: 'rgba(79,70,229,0.2)',
        borderColor: '#4f46e5', borderWidth: 2,
        pointBackgroundColor: '#818cf8', pointRadius: 5,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#94a3b8' } } },
      scales: {
        r: {
          min: 0, max: 100, ticks: { color: '#94a3b8', stepSize: 20 },
          grid: { color: 'rgba(255,255,255,0.07)' },
          angleLines: { color: 'rgba(255,255,255,0.07)' },
          pointLabels: { color: '#94a3b8', font: { size: 12 } }
        }
      }
    }
  });
}

// ─── Form Validation Feedback ────────────────────────
document.querySelectorAll('.form-control-custom').forEach(input => {
  input.addEventListener('input', () => {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const val = parseFloat(input.value);
    if (input.value === '') { input.style.borderColor = ''; return; }
    if (!isNaN(min) && !isNaN(max) && !isNaN(val)) {
      input.style.borderColor = (val >= min && val <= max) ? '#10b981' : '#ef4444';
    }
  });
});

// ─── Confirm Delete ──────────────────────────────────
document.querySelectorAll('.confirm-delete').forEach(btn => {
  btn.addEventListener('click', e => {
    if (!confirm('Are you sure? This action cannot be undone.')) e.preventDefault();
  });
});

// ─── Tooltip init (Bootstrap) ────────────────────────
document.querySelectorAll('[data-bs-toggle="tooltip"]')
  .forEach(el => new bootstrap.Tooltip(el));
