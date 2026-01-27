from pathlib import Path
import requests

OWNER = "KMairot"
REPO = "TestAI"
TAG = "v1.0"
DEST = Path("weights")
DEST.mkdir(parents=True, exist_ok=True)

API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{TAG}"

def main():
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    release = r.json()

    assets = release.get("assets", [])
    pt_assets = [a for a in assets if a.get("name", "").lower().endswith(".pt")]

    if not pt_assets:
        raise RuntimeError("No .pt assets found in the release. Did you attach the checkpoints to v1.0?")

    print(f"Found {len(pt_assets)} checkpoint(s) in release {TAG}:")
    for a in pt_assets:
        print(" -", a["name"])

    for a in pt_assets:
        name = a["name"]
        url = a["browser_download_url"]
        out = DEST / name

        if out.exists() and out.stat().st_size > 0:
            print(f"[OK] {name} already downloaded.")
            continue

        print(f"Downloading {name} ...")
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(out, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"Saved -> {out}")

    print("\nDone. weights/ contains:")
    for p in sorted(DEST.glob("*.pt")):
        print(" -", p.name)

if __name__ == "__main__":
    main()
