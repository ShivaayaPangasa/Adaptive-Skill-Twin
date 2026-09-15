import requests
import zipfile
import io
import time
from pathlib import Path

COLLECTION_ID = 4981073
API_BASE = "https://api.figshare.com/v2"
OUTPUT_DIR = Path("data/raw")


def get_collection_articles(collection_id):
    """Fetch the list of all participant items in the collection, handling pagination."""
    articles = []
    page = 1
    while True:
        url = f"{API_BASE}/collections/{collection_id}/articles"
        resp = requests.get(url, params={"page_size": 100, "page": page})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        articles.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return articles


def get_article_files(article_id):
    """Get the downloadable file(s) for one participant's item."""
    url = f"{API_BASE}/articles/{article_id}/files"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def download_and_extract_zip(download_url, dest_dir):
    resp = requests.get(download_url, stream=True)
    resp.raise_for_status()
    zip_bytes = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_bytes) as zf:
        zf.extractall(dest_dir)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    articles = get_collection_articles(COLLECTION_ID)
    print(f"Found {len(articles)} items in the collection.\n")

    for i, article in enumerate(articles, start=1):
        article_id = article["id"]
        title = article["title"]
        marker = OUTPUT_DIR / f".done_{article_id}"

        print(f"[{i}/{len(articles)}] {title} (id={article_id})")

        if marker.exists():
            print("    already downloaded, skipping")
            continue

        try:
            files = get_article_files(article_id)
        except Exception as e:
            print(f"    could not list files: {e}")
            continue

        for file_info in files:
            filename = file_info["name"]
            print(f"    downloading + extracting {filename} ...")
            try:
                if filename.endswith(".zip"):
                    download_and_extract_zip(file_info["download_url"], OUTPUT_DIR)
                else:
                    dest = OUTPUT_DIR / filename
                    r = requests.get(file_info["download_url"])
                    r.raise_for_status()
                    dest.write_bytes(r.content)
            except Exception as e:
                print(f"    failed: {e}")

        marker.touch()
        time.sleep(0.3)

    print("\nDone.")


if __name__ == "__main__":
    main()