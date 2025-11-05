const form = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const userIdInput = document.querySelector("#user-id-input");
const chatLog = document.querySelector("#chat-log");
const statusIndicator = document.querySelector("#status-indicator");

const SESSION_KEY = "multimodal-chat-history";

function appendMessage(role, text, imageUrls = []) {
  const container = document.createElement("article");
  container.classList.add("message", role);
  container.innerText = text.trim();

  imageUrls.forEach((url) => {
    const img = document.createElement("img");
    img.src = url;
    img.alt = "Generated illustration";
    container.appendChild(img);
  });

  chatLog.appendChild(container);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function persistHistory() {
  const entries = Array.from(chatLog.querySelectorAll(".message")).map((node) => ({
    role: node.classList.contains("user") ? "user" : "assistant",
    text: node.firstChild?.textContent ?? "",
    images: Array.from(node.querySelectorAll("img")).map((img) => img.src),
  }));

  sessionStorage.setItem(SESSION_KEY, JSON.stringify(entries));
}

function hydrateHistory() {
  const raw = sessionStorage.getItem(SESSION_KEY);
  if (!raw) return;

  try {
    const entries = JSON.parse(raw);
    entries.forEach(({ role, text, images }) => appendMessage(role, text, images));
  } catch (err) {
    console.warn("Failed to hydrate chat history", err);
  }
}

async function postChat(message, userId) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, user_id: userId }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || "Chat request failed");
  }

  return response.json();
}

function setStatus(text) {
  statusIndicator.textContent = text;
}

hydrateHistory();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  const userId = userIdInput.value.trim() || "demo-user";
  if (!message) return;

  appendMessage("user", message);
  setStatus("Thinking...");
  form.querySelector("#send-button").disabled = true;

  try {
    const { assistant_response, image_urls = [] } = await postChat(message, userId);
    appendMessage("assistant", assistant_response, image_urls);
    setStatus("Ready");
  } catch (err) {
    console.error(err);
    appendMessage("assistant", `Warning: ${err.message}`);
    setStatus("Error");
  } finally {
    form.reset();
    messageInput.focus();
    form.querySelector("#send-button").disabled = false;
    persistHistory();
  }
});

