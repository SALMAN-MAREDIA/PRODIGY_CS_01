/* ===================================================================
   IPL Score Predictor – Frontend JavaScript
   =================================================================== */

const form       = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const errorAlert = document.getElementById('errorAlert');
const resultCard = document.getElementById('resultCard');

// ---- Prevent same team selection ----
const battingSelect  = document.getElementById('batting_team');
const bowlingSelect  = document.getElementById('bowling_team');

function syncTeamOptions() {
  const selectedBat  = battingSelect.value;
  const selectedBowl = bowlingSelect.value;

  Array.from(bowlingSelect.options).forEach(opt => {
    opt.disabled = opt.value !== '' && opt.value === selectedBat;
  });
  Array.from(battingSelect.options).forEach(opt => {
    opt.disabled = opt.value !== '' && opt.value === selectedBowl;
  });
}

battingSelect.addEventListener('change', syncTeamOptions);
bowlingSelect.addEventListener('change', syncTeamOptions);

// ---- Form submission ----
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideError();

  const payload = {
    batting_team:   battingSelect.value,
    bowling_team:   bowlingSelect.value,
    city:           document.getElementById('city').value,
    overs:          parseFloat(document.getElementById('overs').value),
    runs:           parseInt(document.getElementById('runs').value, 10),
    wickets:        parseInt(document.getElementById('wickets').value, 10),
    runs_last_5:    parseInt(document.getElementById('runs_last_5').value, 10),
    wickets_last_5: parseInt(document.getElementById('wickets_last_5').value, 10),
  };

  // Client-side validation
  if (!payload.batting_team || !payload.bowling_team || !payload.city) {
    return showError('Please select batting team, bowling team, and city.');
  }
  if (payload.batting_team === payload.bowling_team) {
    return showError('Batting and bowling teams must be different.');
  }
  if (isNaN(payload.overs) || payload.overs < 1 || payload.overs > 20) {
    return showError('Overs must be a number between 1 and 20.');
  }
  if (isNaN(payload.runs) || payload.runs < 0 || payload.runs > 400) {
    return showError('Current score must be between 0 and 400.');
  }
  if (isNaN(payload.wickets) || payload.wickets < 0 || payload.wickets > 10) {
    return showError('Wickets must be between 0 and 10.');
  }
  if (isNaN(payload.runs_last_5) || payload.runs_last_5 < 0 || payload.runs_last_5 > 150) {
    return showError('Runs in last 5 overs must be between 0 and 150.');
  }
  if (isNaN(payload.wickets_last_5) || payload.wickets_last_5 < 0 || payload.wickets_last_5 > 10) {
    return showError('Wickets in last 5 overs must be between 0 and 10.');
  }

  // Show loading state
  setLoading(true);

  try {
    const response = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      return showError(data.error || 'Prediction failed. Please try again.');
    }

    displayResult(data, payload);
  } catch (err) {
    showError('Network error. Please check your connection and try again.');
  } finally {
    setLoading(false);
  }
});

// ---- Display result ----
function displayResult(data, payload) {
  document.getElementById('predictedScore').textContent = data.predicted_score;
  document.getElementById('scoreRange').innerHTML =
    `Confidence range: <span class="score-badge bg-primary-subtle text-primary">` +
    `${data.low} – ${data.high} runs</span>`;

  const remainingOvers = Math.max(0, 20 - payload.overs).toFixed(1);
  const currentRR      = payload.overs > 0
    ? (payload.runs / payload.overs).toFixed(2)
    : '—';
  const reqRR = (payload.overs < 20 && remainingOvers > 0)
    ? ((data.predicted_score - payload.runs) / remainingOvers).toFixed(2)
    : '—';

  document.getElementById('matchSummary').innerHTML = `
    <div class="col-4">
      <div class="fw-semibold text-primary">${currentRR}</div>
      <div class="small text-muted">Current RR</div>
    </div>
    <div class="col-4">
      <div class="fw-semibold text-warning">${remainingOvers}</div>
      <div class="small text-muted">Overs Left</div>
    </div>
    <div class="col-4">
      <div class="fw-semibold text-success">${reqRR}</div>
      <div class="small text-muted">Proj. RR</div>
    </div>
  `;

  resultCard.classList.remove('d-none');
  resultCard.classList.add('fade-in-up');
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ---- Helpers ----
function showError(msg) {
  errorAlert.textContent = msg;
  errorAlert.classList.remove('d-none');
  errorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
  errorAlert.classList.add('d-none');
  errorAlert.textContent = '';
}

function setLoading(isLoading) {
  if (isLoading) {
    predictBtn.disabled = true;
    predictBtn.innerHTML =
      '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Predicting…';
  } else {
    predictBtn.disabled = false;
    predictBtn.innerHTML =
      '<i class="bi bi-lightning-charge-fill me-2"></i>Predict Score';
  }
}
