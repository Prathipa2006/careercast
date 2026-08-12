// 
// SHOW / HIDE PASSWORD (login & register pages)
//
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



// 
// FILE UPLOAD (dashboard page)
// 
const upload = document.getElementById("resumeUpload");
const uploadTitle = document.getElementById("uploadTitle");
const uploadSub = document.getElementById("uploadSub");
const uploadBox = document.getElementById("uploadBox");
const analyzeBtn = document.getElementById("analyzeBtn");

let selectedFile = null;

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

// 
// ANALYZING MODAL helpers
// 
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

// 
// ANALYZE BUTTON - real backend call with a clear "before / after" experience
// 
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
        <div class="stat-row">
          <div class="stat">
            <div class="stat-label">MODEL</div>
            <div class="stat-value">Log. Regression</div>
          </div>
          <div class="stat">
            <div class="stat-label">ACCURACY</div>
            <div class="stat-value accent">${data.model_accuracy}%</div>
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
