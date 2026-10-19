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
| `docs/index.html`, `docs/style.css`, `docs/app.js` | Görev 3 – Telegram Mini App arayüzü (GitHub Pages `docs/` klasöründen yayınlar) |
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
.venv\Scriptsctivate
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

> **Skor hakkında:** CLIP'te metin–görsel kosinüs benzerliği doğası gereği düşüktür; iyi eşleşmeler genellikle **%25–35** aralığındadır. Skor mutlak bir "doğruluk" değil, sıralama ölçüsüdür.

### Backend'i HTTPS ile dışarı açma

Telegram Mini App ve GitHub Pages HTTPS olduğu için backend de **HTTPS** olmalıdır (aksi halde tarayıcı isteği engeller). En kolay yol Cloudflare Tunnel:

```bash
cloudflared tunnel --url http://localhost:8010
```

Çıktıdaki `https://xxxx.trycloudflare.com` adresini not edin ve:

1. `.env` → `PUBLIC_BASE_URL=https://xxxx.trycloudflare.com` (görsel linkleri bu adresle üretilir), backend'i yeniden başlatın.
2. `docs/app.js` → `DEFAULT_API_URL` değerini aynı adresle değiştirin.

> Hızlı tünel adresi her başlatmada değişir. Kalıcı kullanım için bir sunucu (VPS, Render, Railway vb.) veya isimli Cloudflare tüneli kullanın.

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

> `docs/app.js` içindeki backend adresini değiştirdiğinizde commit + push yapmanız yeterlidir.
> Tarayıcıda test için: `https://KULLANICI_ADINIZ.github.io/faceid-ai/?api=https://xxxx.trycloudflare.com`

## 5. Bot bağlantısı – BotFather (Görev 4)

1. Telegram'da **@BotFather** → `/newbot` → bot adı ve kullanıcı adı verin, **token**'ı saklayın.
2. Menü butonu olarak bağlama (en basit yol):
   - `/mybots` → botunuzu seçin → **Bot Settings → Menu Button → Configure menu button**
   - URL olarak GitHub Pages adresinizi gönderin: `https://KULLANICI_ADINIZ.github.io/faceid-ai/`
   - Buton yazısı: `Yüz Ara`
3. (İsteğe bağlı) Doğrudan link ile açılan Mini App: BotFather → `/newapp` → botu seçin → başlık, açıklama, 640×360 görsel, Web App URL'si ve kısa ad verin. Uygulama `https://t.me/BOT_KULLANICI_ADI/KISA_AD` linkiyle açılır.
4. (Önerilir) `.env` → `TELEGRAM_BOT_TOKEN=<token>` yazıp backend'i yeniden başlatın. Böylece `/search`, yalnızca botunuzun Mini App'inden gelen ve Telegram imzası doğrulanan istekleri kabul eder; dışarıdan gelen istekler Gemini API kotanızı harcayamaz. (Bu açıkken normal tarayıcıdan arama yapılamaz.)
5. Botu açın, sol alttaki **Yüz Ara** butonuna basın, tarif yazıp **Ara**'ya dokunun.

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| "Sunucuya ulaşılamadı" | Backend ve tünel çalışıyor mu? `app.js` adresi doğru mu? `ALLOWED_ORIGINS` GitHub Pages adresini içeriyor mu? |
| Görseller yüklenmiyor | `PUBLIC_BASE_URL` tünel adresine ayarlı mı? `https://.../health` açılıyor mu? |
| `401 Telegram doğrulaması başarısız` | Mini App'i Telegram içinden açın; `TELEGRAM_BOT_TOKEN` doğru botun token'ı mı? |
| `429 Çeviri servisi yoğun` | Gemini ücretsiz katman limiti doldu; biraz bekleyin veya AI Studio'da faturalandırmayı açın. |
| `502 Çeviri servisi hatası (400/403)` | `GEMINI_API_KEY` eksik veya hatalı. |
| `RuntimeError: Qdrant koleksiyonu yok` | Önce `python process_faces.py` çalıştırın. |
| Qdrant "already accessed by another instance" | `process_faces.py` ve `main.py` aynı anda çalışamaz; birini kapatın. |
