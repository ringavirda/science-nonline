"""Download real-world datasets for validating the dtfit methods.

Two datasets, chosen to match the dissertation's stated domain (nonlinear
smoothing/forecasting on currency / economic / pandemic time series) and the
methods' target shape (models nonlinear in their parameters -- exponential /
transcendental):

1. USD/UAH official exchange rate (National Bank of Ukraine open API).
   Window: the 2014-2015 hryvnia crisis, a sustained, roughly exponential
   depreciation from ~8 to ~30 UAH/USD. Currency-domain test.

2. COVID-19 cumulative confirmed cases (JHU CSSE time series, GitHub).
   The early-2020 growth phase is the textbook exponential-in-parameters
   signal. Pandemic-domain test.

Run:  python experiments/download_data.py
Files land in experiments/data/ as plain CSV (date,value).
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

NBU_URL = (
    "https://bank.gov.ua/NBU_Exchange/exchange_site"
    "?start={start}&end={end}&valcode=usd&sort=exchangedate&order=asc&json"
)
JHU_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)


def _get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dtfit-experiments/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_usd_uah(start: str = "20140101", end: str = "20151231") -> Path:
    """USD/UAH daily official rate over the 2014-2015 hryvnia crisis."""
    raw = _get(NBU_URL.format(start=start, end=end))
    rows = json.loads(raw)
    out = DATA_DIR / "usd_uah_2014_2015.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "rate_uah_per_usd"])
        for r in rows:
            d = datetime.strptime(r["exchangedate"], "%d.%m.%Y").date()
            # NBU historically quoted USD per 100 units; rate_per_unit is the
            # rate for a single USD (e.g. 7.993), which is what we want.
            w.writerow([d.isoformat(), r["rate_per_unit"]])
    print(f"  USD/UAH: {len(rows)} daily rates -> {out.relative_to(DATA_DIR.parent)}")
    return out


def download_covid_ukraine() -> Path:
    """COVID-19 cumulative confirmed cases for Ukraine (JHU CSSE)."""
    raw = _get(JHU_URL).decode("utf-8")
    reader = csv.reader(io.StringIO(raw))
    header = next(reader)
    dates = header[4:]  # columns after Province/State, Country/Region, Lat, Long
    ua = next(row for row in reader if row[1] == "Ukraine")
    values = ua[4:]
    out = DATA_DIR / "covid_ukraine_confirmed.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "cumulative_confirmed"])
        for d, v in zip(dates, values):
            iso = datetime.strptime(d, "%m/%d/%y").date().isoformat()
            w.writerow([iso, int(v)])
    print(f"  COVID UA: {len(dates)} daily points -> {out.relative_to(DATA_DIR.parent)}")
    return out


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading real datasets into {DATA_DIR} ...")
    download_usd_uah()
    download_covid_ukraine()
    print("Done.")


if __name__ == "__main__":
    main()
