const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzone-text");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const generateBtn = document.getElementById("generate-btn");
const resultBox = document.getElementById("result");
const captionText = document.getElementById("caption-text");
const errorBox = document.getElementById("error");

let selectedFile = null;

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropzoneText.hidden = true;
  };
  reader.readAsDataURL(file);

  generateBtn.disabled = false;
  resultBox.hidden = true;
  errorBox.hidden = true;
}

fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  setFile(e.dataTransfer.files[0]);
});

generateBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";
  errorBox.hidden = true;
  resultBox.hidden = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch("/api/caption", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    captionText.textContent = data.caption;
    resultBox.hidden = false;
  } catch (err) {
    errorBox.textContent = err.message || "Something went wrong.";
    errorBox.hidden = false;
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate Caption";
  }
});
