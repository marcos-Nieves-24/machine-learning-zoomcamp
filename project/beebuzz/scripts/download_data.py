"""Download the BeeBuzz dataset (Mendeley, CC BY 4.0).

The dataset is hosted on Mendeley Data (DOI: 10.17632/ggdyfh78d7.1).
The CSV with features is also available in the companion GitHub repo.
"""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/Caarvelos/Bee-acoustics-classification/main/data/buzz_data.csv"


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "buzz_data.csv"
    if dest.exists():
        print(f"skip (exists): {dest}")
        return
    print("downloading buzz_data.csv ...")
    urllib.request.urlretrieve(URL, dest)
    print(f"  -> {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
