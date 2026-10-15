"""Download Creative Commons photos of each species from iNaturalist.

Only research-grade observations (identity confirmed by the community) from
the eastern US, under CC0 / CC-BY / CC-BY-NC licenses. Every photo's credit is
written to attribution.csv, which the licenses require you to keep.

    python training/fetch_inat.py --per-species 200

Output: data/training/raw/<Common Name>/<photo id>.jpg
"""

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

API = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "feeder-cam/0.1 (student bird-feeder ML project)"}
LICENSES = "cc0,cc-by,cc-by-nc"
EASTERN_US = {"swlat": 24, "swlng": -92, "nelat": 47, "nelng": -66}


def api_get(path, **params):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    time.sleep(1.0)  # iNaturalist asks API users to stay around 1 request/second
    return data


def find_taxon(common_name):
    results = api_get("taxa", q=common_name, rank="species", per_page=10)["results"]
    for t in results:
        if (t.get("preferred_common_name") or "").lower() == common_name.lower():
            return t
    raise LookupError(f"no exact iNaturalist match for {common_name!r}; "
                      f"closest: {[t.get('preferred_common_name') for t in results[:3]]}")


def photo_records(taxon_id, want):
    """Yield one photo per observation, so we don't get 5 shots of the same bird."""
    page, seen = 1, 0
    while seen < want:
        data = api_get("observations", taxon_id=taxon_id, quality_grade="research",
                       photos="true", photo_license=LICENSES, per_page=200, page=page,
                       **EASTERN_US)
        if not data["results"]:
            return
        for obs in data["results"]:
            photo = next((p for p in obs["photos"] if p.get("license_code") in LICENSES.split(",")), None)
            if photo is None:
                continue
            seen += 1
            yield obs, photo
            if seen >= want:
                return
        page += 1


def download(url, dest):
    if dest.exists():
        return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as e:  # one failed photo shouldn't stop the run
        print(f"  failed {url}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default=ROOT / "training" / "species_nc.txt", type=Path)
    ap.add_argument("--per-species", type=int, default=200)
    ap.add_argument("--out", default=ROOT / "data" / "training", type=Path)
    args = ap.parse_args()

    names = [l.strip() for l in args.species.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    raw = args.out / "raw"
    attribution_path = args.out / "attribution.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    new_file = not attribution_path.exists()

    with open(attribution_path, "a", newline="") as f, ThreadPoolExecutor(4) as pool:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["species", "file", "photo_id", "license", "attribution", "observation_url"])
        for name in names:
            taxon = find_taxon(name)
            folder = raw / name
            folder.mkdir(parents=True, exist_ok=True)
            jobs = []
            for obs, photo in photo_records(taxon["id"], args.per_species):
                dest = folder / f"{photo['id']}.jpg"
                if dest.exists():  # already downloaded (and credited) on an earlier run
                    continue
                url = photo["url"].replace("/square.", "/medium.")  # ~500px on the long side
                row = [name, dest.relative_to(args.out).as_posix(), photo["id"],
                       photo["license_code"], photo["attribution"], obs["uri"]]
                jobs.append((pool.submit(download, url, dest), row))
            for job, row in jobs:
                if job.result():
                    writer.writerow(row)
            print(f"{name} ({taxon['name']}): {len(list(folder.glob('*.jpg')))} photos")


if __name__ == "__main__":
    main()
