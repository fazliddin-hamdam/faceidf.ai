# Face ID AI: Semantic Face Search System

Türkçe yazılan fiziksel tarifleri (ör. *"esmer, sakallı, gözlüklü"*) Gemini ile İngilizce CLIP sorgusuna çevirir, CLIP vektörüyle LFW veri setinde anlamsal arama yapar ve en yakın yüzü Telegram Mini App içinde gösterir.

```
Telegram Mini App (GitHub Pages)
        │  POST /search {"text": "esmer, sakallı"}
        ▼
FastAPI (main.py) ──► Gemini API: "a photo of a dark haired, bearded man"
        │
        ├──► CLIP clip-ViT-B-32: metin → 512 boyutlu vektör
        └──► Qdrant (yerel, cosine) → en yakın görsel + skor
```

## Dosyalar

| Dosya | Görev |
|---|---|
| `process_faces.py` | Görev 1 – LFW'yi indirir, görselleri CLIP ile vektörleştirip Qdrant'a yazar (her kayıt UUID + `path` payload) |
| `main.py` | Görev 2 – FastAPI backend: `/search`, Gemini çevirisi, CLIP + Qdrant araması, CORS, `/images` |
| `translation_cache.py` | Gemini çevirilerini kalıcı olarak saklayan küçük TTL/LRU önbelleği |
| `docs/index.html`, `docs/style.css`, `docs/app.js` | Görev 3 – Telegram Mini App arayüzü (GitHub Pages `docs/` klasöründen yayınlar) |
| `tests/`, `pyproject.toml`, `.github/workflows/ci.yml` | Testler, kod kontrolü ve GitHub Actions CI |
| `.env.example` | Ortam değişkenleri şablonu |

---

## 1. Kurulum (yerel)

Python 3.11+ gerekir.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # Linux/macOS: cp .env.example .env
```

`.env` dosyasında en az `GEMINI_API_KEY` değerini doldurun. Anahtarı ücretsiz olarak [Google AI Studio](https://aistudio.google.com/apikey) üzerinden alabilirsiniz (Google hesabıyla giriş → **Create API key**).

> Varsayılan model `gemini-3.5-flash-lite` ücretsiz katmanda kullanılabilir. Ücretsiz katmanda dakikalık/günlük istek limiti vardır ve Google gönderilen metinleri ürün geliştirmede kullanabilir; ücretli katmanda (faturalandırma açık proje) bu yapılmaz. Model `.env` → `GEMINI_MODEL` ile değiştirilebilir.

## 2. Veri hazırlığı (Görev 1)

```bash
python process_faces.py --download
```

- LFW (~240 MB, ~13.000 görsel) `data/lfw_funneled/` klasörüne iner ve açılır.
- Tüm görseller CLIP ile vektörleştirilir ve `qdrant_data/` içine kaydedilir. CPU'da 10–30 dk sürebilir; hızlı deneme için `--limit 500` kullanın.
- Tekrar çalıştırmak güvenlidir: UUID'ler dosya yolundan türetildiği için kayıtlar çoğalmaz. Sıfırdan başlamak için `--recreate`.

> Qdrant yerel modda klasörü kilitler: `process_faces.py` çalışırken `main.py` kapalı olmalıdır.

## 3. Backend'i çalıştırma (Görev 2)

Paketler `.venv` içinde kurulu olduğu için önce sanal ortamı etkinleştirin, sonra çalıştırın:

```bash
.venv\Scripts\activate
```

```bash
python main.py
```

Backend `http://127.0.0.1:8010` adresinde açılır (port `.env` → `PORT` ile değiştirilebilir).

> `ModuleNotFoundError: No module named 'fastapi'` hatası alırsanız `.venv` etkin değildir; satır başında `(.venv)` yazısını görmelisiniz.

Test:

```bash
curl -X POST http://localhost:8010/search -H "Content-Type: application/json" -d "{\"text\": \"esmer, sakallı\", \"top_k\": 3}"
```

Örnek yanıt:

```json
{
  "query": "esmer, sakallı",
  "clip_query": "a photo of a dark haired, bearded man",
  "results": [
    {
      "image_url": "http://localhost:8010/images/Some_Person/Some_Person_0001.jpg",
      "path": "Some_Person/Some_Person_0001.jpg",
      "person": "Some Person",
      "score": 0.3121,
      "similarity_percent": 31.2
    }
  ]
}
```

> **Skor hakkında:** CLIP'te metin–görsel kosinüs benzerliği doğası gereği düşüktür; iyi eşleşmeler genellikle **0.25–0.35** aralığındadır. Skor mutlak bir "doğruluk" değil, sıralama ölçüsüdür. `MIN_SCORE` (varsayılan `0.22`) altındaki sonuçlar gösterilmez; aynı kişinin birden fazla fotoğrafı tek sonuçta birleştirilir.

> **Önbellek:** Gemini'ye giden her çeviri `data/translation_cache.json` dosyasına yazılır. Aynı tarif tekrar gelirse Gemini'ye gitmeden yanıtlanır; ücretsiz kota korunur.

### Backend'i HTTPS ile dışarı açma

Telegram Mini App ve GitHub Pages HTTPS olduğu için backend de **HTTPS** olmalıdır (aksi halde tarayıcı isteği engeller).

**Önce şunu bilin:** `cloudflared tunnel --url ...` komutu ücretsiz bir **geçici** tünel açar ve adresi her başlatmada değişir. Bu yüzden `docs/app.js` içindeki adresi sürekli güncellemek zorunda kalırsınız.

Ücretsiz ve sabit adres için üç seçenek:

> Kurulum linkleri: [Tailscale](https://tailscale.com/download/windows) · [ngrok](https://ngrok.com/download). Tailscale yönetici izni ister; ngrok tek dosyadır ama ücretsiz hesap/authtoken gerekir.

1. **Tailscale Funnel (en kolay):**
   ```bash
   tailscale up
   tailscale funnel 8010
   ```
   Size `https://BILGISAYAR_ADI.TAILNET.ts.net` gibi sabit bir adres verir. (Tailscale yönetim panelinden Funnel özelliğini açmanız gerekebilir.)

2. **ngrok ücretsiz statik alan adı:**
   ```bash
   ngrok config add-authtoken <TOKEN>
   ngrok http --url=SIZIN_ADINIZ.ngrok-free.app 8010
   ```
   Ücretsiz planda 1 statik alan adı vardır; bazı istek limitleri olabilir.

3. **Cloudflare named tunnel:** Kendi alan adınız varsa `cloudflared tunnel login` → `cloudflared tunnel create` → `cloudflared tunnel route dns` adımlarıyla kalıcı tünel kurulur. Tünel ücretsizdir; alan adı gerekir.

Adresi bir kez `docs/app.js` içindeki `API_URL` satırına yazın. Dosyada şu an `https://SIZIN-SABIT-TUNEL-ADRESINIZ` örneği durur; değiştirene kadar arayüz "API adresi ayarlanmadı" uyarısı verir. **`?api=` ile adres değiştirme özelliği güvenlik nedeniyle kaldırıldı.**

> Tailscale/ngrok kullanırsanız `.env` → `PUBLIC_BASE_URL=https://sizin-adresiniz` doldurun; görsel linkleri doğru çıksın.

### Hızlı tünel ile devam edecekseniz

```bash
cloudflared tunnel --url http://localhost:8010
```

Çıktıdaki `https://xxxx.trycloudflare.com` adresini `docs/app.js` → `API_URL` satırına yazın. Tünel her açılışta değişeceği için bu adresi her seferinde güncellemeniz gerekir. Bilgisayarın uyumasını kapatın ve sunucu/tüneli başlatmak için bir `.bat` dosyasını `shell:startup` klasörüne koyun.

## 4. Frontend'i GitHub Pages'a yükleme (Görev 4)

1. GitHub'da yeni bir repo açın (ör. `faceid-ai`).
2. Projeyi gönderin (`.gitignore` sayesinde `.env`, `data/`, `qdrant_data/` gönderilmez):
   ```bash
   git init
   git add .
   git commit -m "Face ID AI"
   git branch -M main
   git remote add origin https://github.com/KULLANICI_ADINIZ/faceid-ai.git
   git push -u origin main
   ```
3. Repo → **Settings → Pages → Build and deployment**: Source = *Deploy from a branch*, Branch = `main`, klasör = **`/docs`** → *Save*.
4. 1–2 dakika sonra site `https://KULLANICI_ADINIZ.github.io/faceid-ai/` adresinde yayında olur.
5. `.env` → `ALLOWED_ORIGINS=https://KULLANICI_ADINIZ.github.io` yapın ve backend'i yeniden başlatın (CORS).

> `docs/app.js` içindeki `API_URL` satırını değiştirdiğinizde commit + push yapmanız yeterlidir.
> Tarayıcıda test için sayfayı doğrudan açın: `https://KULLANICI_ADINIZ.github.io/faceid-ai/` (`TELEGRAM_BOT_TOKEN` doluysa arama için Telegram içinden açmanız gerekir).

## 5. Bot bağlantısı – BotFather (Görev 4)

1. Telegram'da **@BotFather** → `/newbot` → bot adı ve kullanıcı adı verin, **token**'ı saklayın.
2. Menü butonu olarak bağlama (en basit yol):
   - `/mybots` → botunuzu seçin → **Bot Settings → Menu Button → Configure menu button**
   - URL olarak GitHub Pages adresinizi gönderin: `https://KULLANICI_ADINIZ.github.io/faceid-ai/`
   - Buton yazısı: `Yüz Ara`
3. (İsteğe bağlı) Doğrudan link ile açılan Mini App: BotFather → `/newapp` → botu seçin → başlık, açıklama, 640×360 görsel, Web App URL'si ve kısa ad verin. Uygulama `https://t.me/BOT_KULLANICI_ADI/KISA_AD` linkiyle açılır.
4. (Önerilir) `.env` → `TELEGRAM_BOT_TOKEN=<token>` yazıp backend'i yeniden başlatın. Böylece `/search`, yalnızca botunuzun Mini App'inden gelen ve Telegram imzası doğrulanan istekleri kabul eder; dışarıdan gelen istekler Gemini API kotanızı harcayamaz. (Bu açıkken normal tarayıcıdan arama yapılamaz.)
5. Botu açın, sol alttaki **Yüz Ara** butonuna basın, tarif yazıp **Ara**'ya dokunun.

## 6. Testler ve kod kontrolü

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

- Testler modeli/Qdrant'ı yüklemez; hızlı çalışır.
- GitHub Actions her push ve pull request'te `ruff`, `compileall` ve `pytest` çalıştırır: `.github/workflows/ci.yml`.
- Yeni bir hata düzeltirken önce testi yazıp sonra kodu düzeltmek en sağlıklısı.

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| "Sunucuya ulaşılamadı" | Backend ve tünel çalışıyor mu? `app.js` adres doğru mu? `ALLOWED_ORIGINS` GitHub Pages adresini içeriyor mu? |
| Görseller yüklenmiyor | `https://.../health` açılıyor mu? Tailscale/ngrok kullanıyorsanız `PUBLIC_BASE_URL` güncel mi? |
| `401 Telegram doğrulaması başarısız` | Mini App'i Telegram içinden açın; `TELEGRAM_BOT_TOKEN` doğru botun token'ı mı? |
| `429 Çeviri servisi yoğun` | Gemini ücretsiz katman limiti doldu; biraz bekleyin veya AI Studio'da faturalandırmayı açın. |
| `502 Çeviri servisi hatası (400/403)` | `GEMINI_API_KEY` eksik veya hatalı. |
| `504 Çeviri servisi zaman aşımına uğradı` | Gemini yavaş yanıtladı; `TRANSLATION_TIMEOUT` değerini artırın veya tekrar deneyin. |
| Sonuç çıkmıyor | Tüm sonuçlar `MIN_SCORE` altında kalmış olabilir; değeri düşürüp deneyin (ör. `0.18`). |
| `RuntimeError: Qdrant koleksiyonu yok` | Önce `python process_faces.py` çalıştırın. |
| Qdrant "already accessed by another instance" | `process_faces.py` ve `main.py` aynı anda çalışamaz; birini kapatın. |
