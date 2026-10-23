"""LFW yüz görsellerini CLIP ile vektörleştirip yerel Qdrant veritabanına kaydeder.

Kullanım:
    python process_faces.py --download          # LFW'yi indir + işle
    python process_faces.py                     # zaten indirilmiş klasörü işle
    python process_faces.py --limit 500         # hızlı deneme için ilk 500 görsel
    python process_faces.py --recreate          # koleksiyonu silip baştan oluştur

Not: Qdrant yerel modda klasörü kilitler; bu betik çalışırken main.py kapalı olmalı.
"""

import argparse
import os
import tarfile
import urllib.request
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_NAME = "clip-ViT-B-32"
VECTOR_SIZE = 512
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# scikit-learn'ün fetch_lfw_people fonksiyonunun kullandığı ayna (~240 MB, hizalanmış "funneled" sürüm)
LFW_URL = "https://ndownloader.figshare.com/files/5976015"
LFW_ARCHIVE_NAME = "lfw-funneled.tgz"


def download_lfw(data_root: Path) -> None:
    from tqdm import tqdm

    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / LFW_ARCHIVE_NAME

    if not archive.exists():
        print(f"LFW indiriliyor: {LFW_URL}")
        request = urllib.request.Request(LFW_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, open(archive, "wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            with tqdm(total=total, unit="B", unit_scale=True) as bar:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    bar.update(len(chunk))

    print("Arşiv açılıyor...")
    with tarfile.open(archive) as tar:
        tar.extractall(data_root, filter="data")


def find_images(lfw_dir: Path) -> list[Path]:
    return sorted(p for p in lfw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def point_id(relative_path: str) -> str:
    # Yola bağlı deterministik UUID: betik tekrar çalıştırılırsa kayıtlar çoğalmaz, güncellenir.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, relative_path))


def ensure_collection(client: QdrantClient, name: str, recreate: bool) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def main() -> None:
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="LFW -> CLIP -> Qdrant")
    parser.add_argument("--lfw-dir", default=os.getenv("LFW_DIR", "data/lfw_funneled"))
    parser.add_argument("--qdrant-path", default=os.getenv("QDRANT_PATH", "qdrant_data"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "faces"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0 = tüm görseller")
    parser.add_argument("--download", action="store_true", help="LFW'yi indir ve aç")
    parser.add_argument("--recreate", action="store_true", help="Koleksiyonu sıfırla")
    args = parser.parse_args()

    lfw_dir = Path(args.lfw_dir)
    if args.download:
        download_lfw(lfw_dir.parent)
    if not lfw_dir.is_dir():
        raise SystemExit(f"LFW klasörü bulunamadı: {lfw_dir} (--download ile indirebilirsiniz)")

    images = find_images(lfw_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"{lfw_dir} içinde görsel bulunamadı")
    print(f"{len(images)} görsel bulundu.")

    print(f"Model yükleniyor: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = QdrantClient(path=args.qdrant_path)
    ensure_collection(client, args.collection, args.recreate)

    skipped = 0
    for start in tqdm(range(0, len(images), args.batch_size), desc="Vektörleştirme"):
        batch_paths = images[start : start + args.batch_size]
        pil_images, payloads = [], []
        for path in batch_paths:
            try:
                with Image.open(path) as img:
                    pil_images.append(img.convert("RGB"))
            except OSError:
                skipped += 1
                continue
            relative = path.relative_to(lfw_dir).as_posix()
            payloads.append({
                "path": relative,
                "person": path.parent.name.replace("_", " "),
            })

        if not pil_images:
            continue

        vectors = model.encode(
            pil_images,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        client.upsert(
            collection_name=args.collection,
            points=[
                PointStruct(id=point_id(p["path"]), vector=v.tolist(), payload=p)
                for p, v in zip(payloads, vectors)
            ],
        )

    count = client.count(args.collection).count
    client.close()
    print(f"Tamamlandı. Koleksiyondaki kayıt sayısı: {count} (atlanan: {skipped})")


if __name__ == "__main__":
    main()
