// ---------- tab switching ----------
const tabs = document.querySelectorAll(".tab");
const modeField = document.getElementById("mode-field");

// Fields inside a hidden (display:none) tab-panel are still part of the form.
// If they keep `required`, the browser tries to validate them on submit,
// can't focus a hidden field to show the error, and silently blocks the
// entire submission. So we toggle `required` on/off based on which tab is active.
const manualInputs = document.querySelectorAll("#panel-manual input");
const csvInput = document.getElementById("csv-file");

function setActiveTab(target) {
    tabs.forEach((t) => t.classList.toggle("active", t.dataset.tab === target));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    document.getElementById(`panel-${target}`).classList.add("active");
    modeField.value = target;

    const isManual = target === "manual";
    manualInputs.forEach((input) => { input.required = isManual; });
    csvInput.required = !isManual;
}

tabs.forEach((tab) => {
    tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
});

// initialize on page load (manual tab is active by default)
setActiveTab("manual");

// ---------- csv filename preview ----------
const dropzoneText = document.getElementById("dropzone-text");
csvInput.addEventListener("change", () => {
    dropzoneText.textContent = csvInput.files.length
        ? csvInput.files[0].name
        : "فایل CSV را انتخاب یا رها کنید";
});

// ---------- form submit ----------
const form = document.getElementById("predict-form");
const submitBtn = document.getElementById("submit-btn");
const btnText = submitBtn.querySelector(".btn-text");
const btnSpinner = submitBtn.querySelector(".btn-spinner");
const errorBox = document.getElementById("form-error");

const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");

function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    btnSpinner.hidden = !isLoading;
    btnText.textContent = isLoading ? "در حال محاسبه..." : "دریافت پیش‌بینی فردا";
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
}

function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
}

function renderResult(data) {
    resultEmpty.hidden = true;
    resultContent.hidden = false;

    const badge = document.getElementById("class-badge");
    badge.className = `class-badge ${data.class_meta.level}`;
    document.getElementById("class-label").textContent = data.class_meta.label;
    document.getElementById("class-msg").textContent = data.alert_message;

    const pct = data.frost_risk_probability_pct;
    const gauge = document.getElementById("gauge");
    const colorVar =
        data.class_meta.level === "safe" ? "#4caf7d" :
        data.class_meta.level === "danger" ? "#e6485a" : "#f0a202";
    gauge.style.setProperty("--pct", pct);
    gauge.style.setProperty("--color", colorVar);
    document.getElementById("gauge-value").textContent = `${pct}%`;

    const t = data.predicted_thermodynamics;
    document.getElementById("m-tmin").textContent = `${t.tmin}°C`;
    document.getElementById("m-tmax").textContent = `${t.tmax}°C`;
    document.getElementById("m-tdew").textContent = `${t.tdew}°C`;
    document.getElementById("m-wind").textContent = `${t.wind} m/s`;

    document.getElementById("result-target").textContent = `بازه زمانی پیش‌بینی: ${data.date_target}`;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();
    setLoading(true);

    try {
        const formData = new FormData(form);
        const res = await fetch("/predict", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (!res.ok) {
            showError(data.error || "خطای نامشخص رخ داد.");
            return;
        }

        renderResult(data);
    } catch (err) {
        showError("ارتباط با سرور برقرار نشد. اتصال شبکه را بررسی کنید.");
    } finally {
        setLoading(false);
    }
});