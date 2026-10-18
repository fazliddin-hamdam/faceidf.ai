// Backend (FastAPI) adresi. Tünel/sunucu adresiniz değişince burayı güncelleyin.
// Test için sayfa URL'sine ?api=https://... eklenerek de geçici olarak değiştirilebilir.
const DEFAULT_API_URL = "https://cocktail-expand-harley-dude.trycloudflare.com";
const API_URL = (new URLSearchParams(location.search).get("api") || DEFAULT_API_URL).replace(/\/+$/, "");

const tg = window.Telegram?.WebApp;
const form = document.getElementById("search-form");
const input = document.getElementById("query");
const button = document.getElementById("search-btn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

if (tg) {
  tg.ready();
  tg.expand();
  if (tg.initData) document.documentElement.classList.add("tg");
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function setLoading(loading) {
  button.disabled = loading;
  button.textContent = loading ? "Aranıyor..." : "Ara";
}

function renderResults(data) {
  const [best, ...others] = data.results;
  if (!best) {
    resultEl.hidden = true;
    setStatus("Eşleşme bulunamadı.");
    return;
  }

  document.getElementById("best-img").src = best.image_url;
  document.getElementById("best-person").textContent = best.person;
  document.getElementById("best-score").textContent = `%${best.similarity_percent}`;
  document.getElementById("clip-query").textContent = data.clip_query;

  const othersEl = document.getElementById("others");
  othersEl.replaceChildren(
    ...others.map((r) => {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = r.image_url;
      img.alt = r.person;
      img.loading = "lazy";
      const cap = document.createElement("figcaption");
      const name = document.createElement("b");
      name.textContent = r.person;
      cap.append(name, `%${r.similarity_percent}`);
      fig.append(img, cap);
      return fig;
    })
  );

  resultEl.hidden = false;
  setStatus("");
}

async function search(text) {
  const response = await fetch(`${API_URL}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Telegram-Init-Data": tg?.initData || "",
    },
    body: JSON.stringify({ text, top_k: 4 }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : `Sunucu hatası (${response.status})`;
    throw new Error(detail);
  }
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (text.length < 2) {
    setStatus("Lütfen birkaç kelimeyle tarif edin.", true);
    return;
  }

  input.blur();
  setLoading(true);
  setStatus("Tarif çevriliyor ve yüzler taranıyor...");
  try {
    renderResults(await search(text));
    tg?.HapticFeedback?.notificationOccurred("success");
  } catch (err) {
    resultEl.hidden = true;
    setStatus(err instanceof TypeError ? "Sunucuya ulaşılamadı. Backend çalışıyor mu?" : err.message, true);
    tg?.HapticFeedback?.notificationOccurred("error");
  } finally {
    setLoading(false);
  }
});

// Enter ile ara, Shift+Enter ile yeni satır
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
