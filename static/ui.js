const form = document.getElementById("prompt-form");
const statusEl = document.getElementById("status");
const resultImage = document.getElementById("result-image");
const resultCaption = document.getElementById("result-caption");
const submitBtn = document.getElementById("submit-btn");

const entriesToPayload = (formData) => {
  const payload = {};
  for (const [key, value] of formData.entries()) {
    if (value === "" || value === null) {
      continue;
    }

    if (["width", "height", "num_inference_steps", "seed"].includes(key)) {
      payload[key] = Number(value);
    } else if (key === "guidance_scale") {
      payload[key] = parseFloat(value);
    } else {
      payload[key] = value;
    }
  }
  return payload;
};

const setLoading = (isLoading) => {
  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? "Generating…" : "Generate Image";
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = entriesToPayload(formData);

  if (!payload.prompt) {
    statusEl.textContent = "Please enter a prompt.";
    return;
  }

  setLoading(true);
  statusEl.textContent = "Contacting Stable Diffusion backend…";
  resultCaption.textContent = "Waiting for the server to answer.";

  try {
    const response = await fetch("/image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      const detail = problem.detail || response.statusText;
      throw new Error(detail);
    }

    const data = await response.json();
    if (Array.isArray(data.urls) && data.urls.length > 0) {
      resultImage.src = data.urls[0];
      const usedMocks = data.used_mocks ? "mocked result" : "generated image";
      resultCaption.textContent = `${usedMocks} for prompt: “${payload.prompt}”`;
      statusEl.textContent = `Job ${data.job_id} completed via ${data.provider}.`;
    } else {
      throw new Error("The server did not provide an image URL.");
    }
  } catch (error) {
    console.error(error);
    statusEl.textContent = `Error: ${error.message}`;
    resultCaption.textContent = "Please try again or inspect the server logs.";
  } finally {
    setLoading(false);
  }
});
