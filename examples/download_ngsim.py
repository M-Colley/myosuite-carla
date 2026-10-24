"""Download NGSIM vehicle trajectories from the US DOT open data portal.

The Next Generation Simulation (NGSIM) trajectories (Federal Highway
Administration, dataset ``8ect-6jqj`` on data.transportation.gov) are licensed
CC BY-SA 3.0. This script requests one location and vehicle class through the
Socrata query API in ordered pages, writes a CSV and a manifest with the exact
query, row count and SHA-256. Keep the data outside version control; the
repository commits only code and aggregate results.

Attribution: U.S. Department of Transportation Federal Highway Administration,
Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data.
https://doi.org/10.21949/1504477
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ENDPOINT = "https://data.transportation.gov/resource/8ect-6jqj.csv"
COLUMNS = (
    "vehicle_id",
    "frame_id",
    "global_time",
    "local_x",
    "local_y",
    "v_length",
    "v_vel",
    "v_acc",
    "lane_id",
    "preceding",
    "space_headway",
)


def _page(where: str, offset: int, limit: int) -> list[list[str]]:
    query = urllib.parse.urlencode(
        {
            "$select": ",".join(COLUMNS),
            "$where": where,
            "$order": "global_time,vehicle_id",
            "$limit": str(limit),
            "$offset": str(offset),
        }
    )
    with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=300) as response:
        rows = list(csv.reader(io.TextIOWrapper(response, encoding="utf-8")))
    if not rows or rows[0] != list(COLUMNS):
        raise RuntimeError(f"Unexpected header at offset {offset}: {rows[:1]}")
    return rows[1:]


def main() -> int:
    """Download the selected subset and write its manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", default="i-80")
    parser.add_argument("--vehicle-class", type=int, default=2, help="2 = cars")
    parser.add_argument("--page", type=int, default=500_000)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/ngsim/ngsim-i80-cars.csv")
    )
    args = parser.parse_args()
    where = f"location='{args.location}' AND v_class={args.vehicle_class}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        digest.update((",".join(COLUMNS) + "\r\n").encode())
        while True:
            rows = _page(where, total, args.page)
            writer.writerows(rows)
            for row in rows:
                digest.update((",".join(row) + "\r\n").encode())
            total += len(rows)
            print(f"{total} rows", flush=True)
            if len(rows) < args.page:
                break
    manifest = {
        "source": "https://data.transportation.gov/d/8ect-6jqj",
        "license": "CC BY-SA 3.0",
        "attribution": (
            "U.S. Department of Transportation Federal Highway Administration, "
            "Next Generation Simulation (NGSIM) Vehicle Trajectories and "
            "Supporting Data, https://doi.org/10.21949/1504477"
        ),
        "endpoint": ENDPOINT,
        "select": list(COLUMNS),
        "where": where,
        "order": "global_time,vehicle_id",
        "rows": total,
        "sha256_of_rows": digest.hexdigest(),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
