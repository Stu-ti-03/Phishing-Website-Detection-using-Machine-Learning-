/**
 * script.js
 * ---------
 * Frontend logic for PhishGuard.
 * Calls POST /predict and updates the UI dynamically.
 */

// ── DOM references ──────────────────────────────────────────────────────────
const urlInput    = document.getElementById('urlInput');
const checkBtn    = document.getElementById('checkBtn');
const loader      = document.getElementById('loader');
const resultCard  = document.getElementById('resultCard');
const errorMsg    = document.getElementById('errorMsg');

// Result elements
const verdictBanner = document.getElementById('verdictBanner');
const verdictIcon   = document.getElementById('verdictIcon');
const verdictLabel  = document.getElementById('verdictLabel');
const detailUrl     = document.getElementById('detailUrl');
const probFill      = document.getElementById('probFill');
const probPct       = document.getElementById('probPct');
const detailRule    = document.getElementById('detailRule');
const meterFill     = document.getElementById('meterFill');
const meterNeedle   = document.getElementById('meterNeedle');

// ── Verdict config ──────────────────────────────────────────────────────────
const VERDICT_CONFIG = {
  'Phishing': {
    className: 'phishing',
    icon: '⚠',
    label: 'PHISHING DETECTED',
    probColor: '#ff3b5c',
  },
  'Legitimate': {
    className: 'legitimate',
    icon: '✔',
    label: 'LEGITIMATE',
    probColor: '#00e97a',
  },
  'Uncertain': {
    className: 'uncertain',
    icon: '◈',
    label: 'UNCERTAIN — PROCEED WITH CAUTION',
    probColor: '#ffcb30',
  },
  'Invalid Input': {
    className: 'invalid',
    icon: '✖',
    label: 'INVALID INPUT',
    probColor: '#888',
  },
};

// ── Utility helpers ─────────────────────────────────────────────────────────

/** Show an element (remove hidden attribute) */
function show(el) { el.hidden = false; }

/** Hide an element (set hidden attribute) */
function hide(el) { el.hidden = true; }

/** Display an error message */
function showError(msg) {
  errorMsg.textContent = msg;
  show(errorMsg);
}

/** Clear any existing error */
function clearError() {
  hide(errorMsg);
  errorMsg.textContent = '';
}

/** Set the loading state of the button */
function setLoading(isLoading) {
  checkBtn.disabled = isLoading;
  checkBtn.querySelector('.btn-text').textContent = isLoading ? 'Scanning…' : 'Analyse';
  isLoading ? show(loader) : hide(loader);
}

// ── Main analysis function ──────────────────────────────────────────────────

async function analyseURL() {
  const raw = urlInput.value.trim();

  // Client-side guard
  if (!raw) {
    showError('Please enter a URL to analyse.');
    return;
  }

  // Reset UI
  clearError();
  hide(resultCard);
  setLoading(true);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: raw }),
    });

    if (!response.ok) {
      // Try to parse API error detail
      const err = await response.json().catch(() => null);
      throw new Error(err?.detail || `Server error (${response.status})`);
    }

    const data = await response.json();
    renderResult(data);

  } catch (err) {
    showError(`❌ Request failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// ── Render result ───────────────────────────────────────────────────────────

function renderResult(data) {
  const { url, prediction, probability, rule_triggered } = data;
  const cfg = VERDICT_CONFIG[prediction] ?? VERDICT_CONFIG['Invalid Input'];

  // --- Verdict banner ---
  verdictBanner.className = `verdict-banner ${cfg.className}`;
  verdictIcon.textContent  = cfg.icon;
  verdictLabel.textContent = cfg.label;

  // --- URL ---
  detailUrl.textContent = url;

  // --- Probability bar ---
  const pct = Math.round(probability * 100);
  // Animate on next frame so CSS transition fires
  requestAnimationFrame(() => {
    probFill.style.width       = `${pct}%`;
    probFill.style.background  = cfg.probColor;
    probPct.style.color        = cfg.probColor;
  });
  probPct.textContent = `${pct}%`;

  // --- Rule triggered ---
  const ruleClean = rule_triggered ?? 'None';
  detailRule.textContent = ruleClean;
  if (ruleClean === 'None') {
    detailRule.className = 'detail-val rule-val clear';
  } else {
    detailRule.textContent = `⚑ ${ruleClean}`;
    detailRule.className   = 'detail-val rule-val triggered';
  }

  // --- Threat meter ---
  // probability is phishing probability (0→safe, 1→danger)
  requestAnimationFrame(() => {
    meterFill.style.transform = `scaleX(${probability})`;
    meterNeedle.style.left    = `${probability * 100}%`;
  });

  // Show the card
  show(resultCard);
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Event listeners ─────────────────────────────────────────────────────────

checkBtn.addEventListener('click', analyseURL);

urlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') analyseURL();
});

// Quick-test pills
document.querySelectorAll('.qt-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    urlInput.value = pill.dataset.url;
    urlInput.focus();
    analyseURL();
  });
});

// Auto-focus input on load
urlInput.focus();