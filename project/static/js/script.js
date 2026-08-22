// ----------------------------
// SHOW / HIDE PASSWORD (login & register pages)
// ----------------------------
const togglePassword = document.getElementById("togglePassword");
const password = document.getElementById("password");

if (togglePassword) {
  togglePassword.addEventListener("click", function () {
    if (password.type === "password") {
      password.type = "text";
      togglePassword.classList.remove("fa-eye");
      togglePassword.classList.add("fa-eye-slash");
    } else {
      password.type = "password";
      togglePassword.classList.remove("fa-eye-slash");
      togglePassword.classList.add("fa-eye");
    }
  });
}

// NOTE: login/register forms submit natively to Flask (method="POST",
// action="/login" or "/register" in the HTML) - no JS needed for that,
// so Flask can validate and show real error messages server-side.

// ----------------------------
// RESTORE last analysis on page load (so navigating to Analytics
// and back doesn't wipe out your results)
// ----------------------------
async function restoreLastAnalysis() {
  const resultBox = document.getElementById("resultBox");
  if (!resultBox) return; // not on the dashboard page

  try {
    const res = await fetch("/api/last-analysis");
    const data = await res.json();
    if (!data) return; // nothing analyzed yet this session

    renderResults(data, resultBox);
    resultBox.classList.remove("empty");

    if (uploadBox && uploadTitle && uploadSub) {
      uploadBox.classList.add("analyzed");
      uploadTitle.textContent = data.filename || "Previous resume";
      uploadSub.textContent = "✓ Showing your last analysis";
    }
  } catch (err) {
    // silently ignore - just leave the placeholder text showing
  }
}

const uploadTitle = document.getElementById("uploadTitle");
const uploadSub = document.getElementById("uploadSub");
const uploadBox = document.getElementById("uploadBox");
const analyzeBtn = document.getElementById("analyzeBtn");
const upload = document.getElementById("resumeUpload");

let selectedFile = null;

restoreLastAnalysis(); // safe now - uploadBox/uploadTitle/uploadSub already declared above

if (upload) {
  upload.addEventListener("change", function () {
    if (upload.files.length > 0) {
      selectedFile = upload.files[0];
      uploadTitle.textContent = selectedFile.name;
      uploadSub.textContent = "File selected - click Analyze Resume";
      analyzeBtn.disabled = false;
      uploadBox.classList.remove("analyzed");
    }
  });

  // drag & drop support
  ["dragenter", "dragover"].forEach(evt =>
    uploadBox.addEventListener(evt, e => { e.preventDefault(); uploadBox.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach(evt =>
    uploadBox.addEventListener(evt, e => { e.preventDefault(); uploadBox.classList.remove("drag"); })
  );
  uploadBox.addEventListener("drop", e => {
    const file = e.dataTransfer.files[0];
    if (file) {
      selectedFile = file;
      uploadTitle.textContent = file.name;
      uploadSub.textContent = "File selected - click Analyze Resume";
      analyzeBtn.disabled = false;
    }
  });
}

// ----------------------------
// ANALYZING MODAL helpers
// ----------------------------
const analyzingModal = document.getElementById("analyzingModal");
const modalSteps = document.getElementById("modalSteps") ? [...document.getElementById("modalSteps").children] : [];

function showModal() {
  if (!analyzingModal) return;
  modalSteps.forEach(li => li.classList.remove("active", "done"));
  analyzingModal.classList.add("show");
}

function hideModal() {
  if (!analyzingModal) return;
  analyzingModal.classList.remove("show");
}

function stepTo(index) {
  // marks steps before `index` as done, and `index` itself as active
  modalSteps.forEach((li, i) => {
    li.classList.remove("active", "done");
    if (i < index) li.classList.add("done");
    else if (i === index) li.classList.add("active");
  });
}

function finishAllSteps() {
  modalSteps.forEach(li => { li.classList.remove("active"); li.classList.add("done"); });
}

// ----------------------------
// ANALYZE BUTTON - real backend call with a clear "before / after" experience
// ----------------------------
if (analyzeBtn) {
  analyzeBtn.addEventListener("click", async function () {
    if (!selectedFile) return;

    const resultBox = document.getElementById("resultBox");
    analyzeBtn.disabled = true;

    // 1) BEFORE state: show the modal and start stepping through stages
    showModal();
    stepTo(0);
    const stepTimer1 = setTimeout(() => stepTo(1), 600);
    const stepTimer2 = setTimeout(() => stepTo(2), 1300);
    const stepTimer3 = setTimeout(() => stepTo(3), 2000);

    const formData = new FormData();
    formData.append("resume", selectedFile);

    try {
      const res = await fetch("/analyze", { method: "POST", body: formData });
      const data = await res.json();

      // make sure the modal shows at least the later steps briefly,
      // even if the server responded very fast, so it doesn't just flash
      const elapsed = performance.now();
      await new Promise(r => setTimeout(r, 400));

      finishAllSteps();
      await new Promise(r => setTimeout(r, 350));

      if (!res.ok) {
        resultBox.innerHTML = `<p class="result-error">⚠️ ${data.error || "Something went wrong."}</p>`;
      } else {
        renderResults(data, resultBox);
        uploadBox.classList.add("analyzed");
        uploadSub.textContent = "✓ Analysis complete";
      }

      // 2) AFTER state: hide modal, scroll to results, pulse-highlight the box
      hideModal();
      resultBox.classList.add("just-updated");
      document.getElementById("resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
      setTimeout(() => resultBox.classList.remove("just-updated"), 1200);

    } catch (err) {
      clearTimeout(stepTimer1); clearTimeout(stepTimer2); clearTimeout(stepTimer3);
      hideModal();
      resultBox.innerHTML = `<p class="result-error">⚠️ Could not reach the server.</p>`;
    } finally {
      analyzeBtn.disabled = false;
    }
  });
}

function renderResults(data, container) {
  currentUserSkills = data.skills || [];  // feed the skill-gap checker below

  const skillsHtml = data.skills.length
    ? data.skills.map(s => `<span class="chip skill-chip">${s}</span>`).join("")
    : `<span class="muted">No recognized skills found</span>`;

  const eduHtml = data.education.length
    ? data.education.map(e => `<span class="chip edu-chip">${e}</span>`).join("")
    : `<span class="muted">No education keywords found</span>`;

  const broadHtml = data.broad_field_predictions
    .map(p => `<span class="chip field-chip">${p.category} · ${p.confidence}%</span>`)
    .join("");

  const rolesHtml = data.role_matches.map(r => `
    <div class="role-row">
      <div class="role-head">
        <span class="role-name">${r.role}</span>
        <span class="role-pct">${r.match_percent}%</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:${r.match_percent}%"></div></div>
      <div class="role-gap">
        ${r.missing_skills.length ? `Gap: <b>${r.missing_skills.slice(0, 3).join(", ")}</b>` : "All core skills matched"}
      </div>
    </div>
  `).join("");

  container.innerHTML = `
    <div class="result-grid result-fade-in">
      <div class="result-col">
        <h4>Extracted Skills</h4>
        <div class="chip-row">${skillsHtml}</div>

        <h4>Education Detected</h4>
        <div class="chip-row">${eduHtml}</div>
      </div>

      <div class="result-col">
        <div class="model-accuracy-section">

    <div class="model-accuracy-title">
        Model Performance
    </div>

    <div class="model-accuracy-grid">

        <div class="model-accuracy-card">
            <div class="kpi-label">Model</div>
            <div class="model-name">Log. Regression</div>
            <div class="kpi-label">Accuracy</div>
            <div class="model-score">${data.model_accuracies?.logistic_regression ?? "78.89"}%</div>
        </div>

        <div class="model-accuracy-card best">
            <div class="kpi-label">Model</div>
            <div class="model-name">Random Forest</div>
            <div class="kpi-label">Accuracy</div>
            <div class="model-score">${data.model_accuracies?.random_forest ?? "80.88"}%</div>
        </div>

        <div class="model-accuracy-card">
            <div class="kpi-label">Model</div>
            <div class="model-name">XGBoost</div>
            <div class="kpi-label">Accuracy</div>
            <div class="model-score">${data.model_accuracies?.xgboost ?? "80.68"}%</div>
        </div>

    </div>

</div>

        <h4>Broad Field Prediction</h4>
        <div class="chip-row">${broadHtml}</div>

        <h4>Specific Role Match</h4>
        ${rolesHtml}

        <div class="verdict">
          <div class="verdict-label">BEST MATCH</div>
          <div class="verdict-text">Best suited for <span>${data.best_role}</span></div>
        </div>
      </div>
    </div>
  `;
}

// ----------------------------
// LOGOUT
// ----------------------------
const logout = document.getElementById("logoutBtn");
if (logout) {
  logout.addEventListener("click", function () {
    window.location.href = "/logout";
  });
}

// ----------------------------
// SKILL GAP FOR A TARGET DOMAIN (mentor's requested feature)
// User picks ANY career, sees exactly what they're missing for it -
// independent of what the model actually predicted.
// ----------------------------
let currentUserSkills = [];  // updated every time a resume is analyzed

const targetRoleSelect = document.getElementById("targetRoleSelect");
const checkGapBtn = document.getElementById("checkGapBtn");
const gapResultBox = document.getElementById("gapResultBox");
const currentSkillsPreview = document.getElementById("currentSkillsPreview");
const noSkillsWarning = document.getElementById("noSkillsWarning");

async function loadCurrentSkillsForGapPage() {
  // Only runs on the dedicated Skill Gap Analysis page
  if (!currentSkillsPreview) return;

  try {
    const res = await fetch("/api/last-analysis");
    const data = await res.json();

    if (!data || !data.skills || data.skills.length === 0) {
      currentSkillsPreview.textContent = "none found yet";
      if (noSkillsWarning) noSkillsWarning.style.display = "block";
      return;
    }

    currentUserSkills = data.skills;
    currentSkillsPreview.textContent = data.skills.join(", ");

    renderPredictedRoleComparison(data);
  } catch (err) {
    currentSkillsPreview.textContent = "could not load";
    console.error("Could not load last analysis for skill gap page:", err);
  }
}

function renderPredictedRoleComparison(data) {
  const box = document.getElementById("predictedGapBox");
  if (!box) return;

  // The model's top predicted role already has matched/missing skills
  // computed by match_roles() in app.py - reuse it directly, no new
  // backend call needed.
  const topMatch = (data.role_matches && data.role_matches[0]) || null;
  if (!topMatch) {
    box.innerHTML = `<p>No role prediction available yet.</p>`;
    return;
  }

  const haveHtml = topMatch.matched_skills.length
    ? topMatch.matched_skills.map(s => `<span class="chip skill-chip">${s}</span>`).join("")
    : `<span class="muted">None matched yet</span>`;

  const needHtml = topMatch.missing_skills.length
    ? topMatch.missing_skills.map(s => `<span class="chip edu-chip">${s}</span>`).join("")
    : `<span class="muted">You already have all core skills for this role!</span>`;

  box.innerHTML = `
    <div class="verdict" style="margin-top:0; margin-bottom:20px;">
      <div class="verdict-label">PREDICTED BEST MATCH</div>
      <div class="verdict-text">${data.best_role} <span>${topMatch.match_percent}% match</span></div>
    </div>
    <div class="compare-grid">
      <div class="compare-col have">
        <h4>✓ Skills You Already Have</h4>
        <div class="chip-row">${haveHtml}</div>
      </div>
      <div class="compare-col need">
        <h4>+ Skills To Add</h4>
        <div class="chip-row">${needHtml}</div>
      </div>
    </div>
  `;
}

async function loadAvailableRoles() {
  if (!targetRoleSelect) return;
  try {
    const res = await fetch("/api/available-roles");

    if (!res.ok) {
      const opt = document.createElement("option");
      opt.textContent = `Error loading roles (status ${res.status}) - try refreshing`;
      targetRoleSelect.appendChild(opt);
      console.error("available-roles fetch failed with status:", res.status);
      return;
    }

    const roles = await res.json();

    if (!Array.isArray(roles) || roles.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "No roles available - check ROLE_SKILL_MAP in app.py";
      targetRoleSelect.appendChild(opt);
      return;
    }

    roles.forEach(role => {
      const opt = document.createElement("option");
      opt.value = role;
      opt.textContent = role;
      targetRoleSelect.appendChild(opt);
    });
  } catch (err) {
    const opt = document.createElement("option");
    opt.textContent = "Could not reach server - check console (F12)";
    targetRoleSelect.appendChild(opt);
    console.error("Could not load available roles:", err);
  }
}

if (targetRoleSelect) {
  targetRoleSelect.addEventListener("change", () => {
    checkGapBtn.disabled = !targetRoleSelect.value;
  });
}

if (checkGapBtn) {
  checkGapBtn.addEventListener("click", async () => {
    const targetRole = targetRoleSelect.value;
    if (!targetRole) return;

    checkGapBtn.disabled = true;
    checkGapBtn.textContent = "Checking...";

    try {
      const res = await fetch("/api/skill-gap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_role: targetRole, skills: currentUserSkills }),
      });
      const data = await res.json();

      if (!res.ok) {
        gapResultBox.innerHTML = `<p class="result-error">⚠️ ${data.error || "Something went wrong."}</p>`;
        return;
      }

      const matchedHtml = data.matched_skills.length
        ? data.matched_skills.map(s => `<span class="chip skill-chip">${s}</span>`).join("")
        : `<span class="muted">None yet</span>`;

      const missingHtml = data.missing_skills.length
        ? data.missing_skills.map(s => `<span class="chip edu-chip">${s}</span>`).join("")
        : `<span class="muted">None - you have all core skills for this role!</span>`;

      const suggestionsHtml = data.suggestions.length
        ? `<ul style="margin-top:10px; padding-left:20px;">${data.suggestions.map(s => `<li style="margin-bottom:6px; font-size:13px; color:var(--text-muted);">${s}</li>`).join("")}</ul>`
        : "";

      gapResultBox.innerHTML = `
        <div class="verdict" style="margin-top:0;">
          <div class="verdict-label">SKILL GAP FOR ${data.target_role.toUpperCase()}</div>
          <div class="verdict-text">${data.match_percent}% match</div>
        </div>
        <h4 style="margin-top:16px;">You already have</h4>
        <div class="chip-row">${matchedHtml}</div>
        <h4 style="margin-top:16px;">You're missing</h4>
        <div class="chip-row">${missingHtml}</div>
        ${suggestionsHtml}
      `;
    } catch (err) {
      gapResultBox.innerHTML = `<p class="result-error">⚠️ Could not reach the server.</p>`;
    } finally {
      checkGapBtn.disabled = false;
      checkGapBtn.textContent = "Check My Skill Gap";
    }
  });
}

loadAvailableRoles();
loadCurrentSkillsForGapPage();

// ----------------------------
// MILESTONE 2 ANALYTICS DASHBOARD (moved here from inline <script>
// for consistency with every other page)
// ----------------------------
const modelComparisonCanvas = document.getElementById("modelComparisonChart");

if (modelComparisonCanvas) {
  const CLUSTER_COLORS = ["#3B82F6", "#EC4899", "#F59E0B", "#8B5CF6", "#10B981", "#EF4444"];
  const PRIMARY = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim() || "#4F46E5";

  async function loadAnalytics() {
    const res = await fetch("/api/analytics");
    const data = await res.json();
    renderModelComparison(data.model_comparison);
    renderTop5(data.top5);
    renderTSNE(data.tsne);
  }

  function renderModelComparison(models) {
    if (!models || models.length === 0) {
      modelComparisonCanvas.parentElement.innerHTML =
        '<p class="muted">No model comparison data yet - run Cell K in your notebook to generate model_comparison.json.</p>';
      return;
    }
    const best = Math.max(...models.map(m => m.macro_f1));
    const colors = models.map(m => m.macro_f1 === best ? "#D97706" : PRIMARY);

    new Chart(modelComparisonCanvas, {
      type: "bar",
      data: {
        labels: models.map(m => m.model),
        datasets: [{ label: "Macro F1-Score", data: models.map(m => m.macro_f1), backgroundColor: colors, borderRadius: 6 }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, max: 1 } }
      }
    });
  }

  function renderTop5(top5) {
    const container = document.getElementById("top5List");
    if (!top5 || top5.length === 0) return;
    container.innerHTML = top5.map((r, i) => `
      <div class="top5-row">
        <span class="rank">${i + 1}</span>
        <span class="role-name">${r.role}</span>
        <span class="confidence">${r.match_percent}%</span>
      </div>
    `).join("");
  }

  function renderTSNE(points) {
    const tsneCanvas = document.getElementById("tsneChart");
    if (!points || points.length === 0) {
      tsneCanvas.parentElement.innerHTML =
        '<p class="muted">No embedding data yet - run Cell K in your notebook to generate tsne_data.json.</p>';
      return;
    }
    const clusters = [...new Set(points.map(p => p.cluster))];
    const datasets = clusters.map((cluster, i) => ({
      label: cluster,
      data: points.filter(p => p.cluster === cluster).map(p => ({ x: p.x, y: p.y, skill: p.skill })),
      backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length],
      pointRadius: 5,
    }));

    new Chart(tsneCanvas, {
      type: "scatter",
      data: { datasets },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => `${ctx.raw.skill} (${ctx.dataset.label})` } }
        },
        scales: {
          x: { title: { display: true, text: "t-SNE Component 1" } },
          y: { title: { display: true, text: "t-SNE Component 2" } },
        }
      }
    });

    document.getElementById("tsneLegend").innerHTML = clusters.map((c, i) => `
      <span class="legend-item"><i style="background:${CLUSTER_COLORS[i % CLUSTER_COLORS.length]}"></i>${c}</span>
    `).join("");
  }

  loadAnalytics();
}
