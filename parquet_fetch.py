import time, os, requests
from urllib.parse import urlparse

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
UA = "NYC-Taxi-Student-Project (contact: ********@********.***)"
CACHE_DIR = "./taxi_parquet_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def year_urls(year):
    return [f"{BASE}/yellow_tripdata_{year}-{m:02d}.parquet" for m in range(1, 13)]


def cache_path_for(url):
    name = os.path.basename(urlparse(url).path)
    year = name.split("_")[-1].split("-")[0]
    d = os.path.join(CACHE_DIR, year)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def head_size(url, timeout=15):
    # HEAD size is used as a lightweight cache integrity check
    r = requests.head(url, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    cl = r.headers.get("Content-Length")
    return int(cl) if cl and cl.isdigit() else None


def download_with_backoff(url, dst_path, max_retries=6, base_sleep=2.0):
    attempt = 0
    while True:
        try:
            with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=30) as r:
                if r.status_code in (403, 429):
                    retry_after = r.headers.get("Retry-After")
                    wait = float(retry_after) if (retry_after and retry_after.isdigit()) else base_sleep * (2 ** attempt) + (0.5 * attempt)
                    attempt += 1
                    if attempt > max_retries:
                        raise RuntimeError(f"Exceeded retries for {url} (HTTP {r.status_code})")
                    print(f"{r.status_code} from server; backing off {wait:.1f}s ...")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                tmp = dst_path + ".part"
                # Write to a temp file first, then atomically move into place
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, dst_path)
                return
        except requests.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = base_sleep * (2 ** attempt) + (0.5 * attempt)
            print(f"Network error {e}; retry in {wait:.1f}s (attempt {attempt}/{max_retries}) ...")
            time.sleep(wait)


def fetch_year_with_throttle(year, min_sleep_between_files=6.0, max_mb_per_min=500):
    total_bytes = 0
    files = []
    for url in year_urls(year):
        dst = cache_path_for(url)
        try:
            size_remote = head_size(url)
        except Exception as e:
            print(f"[HEAD] {url} failed: {e}. Will attempt GET ...")
            size_remote = None

        # If local file size matches HEAD metadata, reuse cache and skip GET
        if os.path.exists(dst):
            size_local = os.path.getsize(dst)
            if size_remote is None or size_local == size_remote:
                print(f"[cache] Using {dst} ({size_local/1e6:.1f} MB)")
                files.append(dst)
                continue

        t0 = time.time()
        print(f"[get ] {url}")
        download_with_backoff(url, dst)
        dt = time.time() - t0
        size_local = os.path.getsize(dst)
        files.append(dst)
        total_bytes += size_local
        print(f"[done] {dst} {size_local/1e6:.1f} MB in {dt:.1f}s")

        # Throttle pace to reduce server pressure and rate-limit risk
        mb = size_local / 1e6
        target_seconds = (mb / max_mb_per_min) * 60.0
        sleep_s = max(min_sleep_between_files, target_seconds)
        print(f"[throttle] sleeping {sleep_s:.1f}s ...")
        time.sleep(sleep_s)

    print(f"Year {year}: downloaded ~{total_bytes/1e9:.2f} GB this run")
    return files


# Adjust year range as needed for full historical pulls
for y in range(2016, 2023):
    parquet_list = fetch_year_with_throttle(y)
