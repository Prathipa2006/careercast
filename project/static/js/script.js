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

// 
// LOGOUT
// 
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

async function loadAvailableRoles() {
  if (!targetRoleSelect) return;
  try {
    const res = await fetch("/api/available-roles");
    const roles = await res.json();
    roles.forEach(role => {
      const opt = document.createElement("option");
      opt.value = role;
      opt.textContent = role;
      targetRoleSelect.appendChild(opt);
    });
  } catch (err) {
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
