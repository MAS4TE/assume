# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Create an input folder for ``simulation.py``.

    # small synthetic example, runs without any data download
    python make_inputs.py synthetic --out inputs/example

    # households from synload-de (output/timeseries.parquet needs pyarrow), prices and PV
    # from the data folder of the battery-utility-calculator
    python make_inputs.py synload --out inputs/synload \\
        --timeseries ~/code/synload-de/output/timeseries.parquet \\
        --prices ~/code/battery-utility-calculator/data/prices.csv \\
        --pv ~/code/battery-utility-calculator/data/pv.parquet \\
        --profiles 0 1 2 3 --sellers 2

For your own load profiles skip this script and write the files yourself, the
format is in GRID_SIM.md.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOAD_COMPONENTS = ["base_load_kw", "hp_load_kw", "dhw_load_kw", "ev_load_kw"]


def write_inputs(out: Path, units: pd.DataFrame, loads: pd.DataFrame, prices: pd.DataFrame,
                 solar: pd.DataFrame | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    units.to_csv(out / "units.csv", index=False)
    loads.to_csv(out / "loads.csv")
    prices.to_csv(out / "prices.csv")
    if solar is not None:
        solar.to_csv(out / "solar.csv")
    print(f"wrote {out}: {len(units)} units, {len(loads)} steps from {loads.index[0]} to {loads.index[-1]}")


def synthetic(args) -> None:
    """Sine shaped loads, PV and prices, only to try the simulation."""
    rng = np.random.default_rng(args.seed)
    index = pd.date_range(args.start, periods=int(args.days * 96), freq="15min", name="datetime")
    hour = index.hour + index.minute / 60
    day = (index - index[0]).total_seconds() / 86400

    n = args.buyers + args.sellers
    loads, solar = {}, {}
    for i in range(n):
        base = 0.35 + 0.15 * rng.random()
        morning = np.exp(-((hour - 7.5) ** 2) / 2)
        evening = 1.5 * np.exp(-((hour - 19) ** 2) / 3)
        loads[f"load_{i}"] = np.clip(base + 0.4 * morning + 0.8 * evening + rng.normal(0, 0.05, len(index)), 0.1, None)
        sun = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * (0.6 + 0.4 * np.sin(day / 7))
        seller = i >= args.buyers
        # sellers: low flat load and no PV, so their storage is cheap to rent out and the buyers' PV makes it valuable
        if seller:
            loads[f"load_{i}"] = loads[f"load_{i}"] * 0.4
        solar[f"pv_{i}"] = sun * 0.0 if seller else sun * (3.0 + 4.0 * rng.random())
    loads = pd.DataFrame(loads, index=index).round(4)
    solar = pd.DataFrame(solar, index=index).round(4)

    # supplier 0.32, community 0.22, EEG 0.07: shifting PV to the evening is worth money, arbitrage on the wholesale price a bit
    wholesale = 0.20 + 0.06 * np.sin((hour - 14) / 24 * 2 * np.pi)
    prices = pd.DataFrame(
        {"wholesale": wholesale, "eeg": 0.07, "grid": 0.32, "community": 0.22}, index=index
    ).round(5)

    rows = []
    for i in range(n):
        seller = i >= args.buyers
        rows.append(
            {
                "unit_id": f"{'S' if seller else 'B'}_{i + 1:02d}",
                "role": "sell" if seller else "buy",
                "location": "aachen",
                "load_profile": f"load_{i}",
                "solar_profile": f"pv_{i}",
                "battery_kwh": 10 if seller else 0,
                "tradeable_pct": 80,
                "risk_tolerance": "Low",
                "trading_scope": "All",
            }
        )
    write_inputs(Path(args.out), pd.DataFrame(rows), loads, prices, solar)


def synload(args) -> None:
    ts = pd.read_parquet(args.timeseries)
    ts["datetime"] = pd.to_datetime(ts["datetime"])
    if ts["datetime"].dt.tz is not None:
        ts["datetime"] = ts["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    profiles = args.profiles
    ts = ts[ts["profile_id"].isin(profiles)]
    loads = (
        ts.assign(load=ts[LOAD_COMPONENTS].sum(axis=1))
        .pivot(index="datetime", columns="profile_id", values="load")
        .rename(columns=lambda c: f"load_{c}")
    )

    prices = pd.read_csv(args.prices, index_col="datetime", parse_dates=True)
    prices = prices.drop(columns=[c for c in prices.columns if c.startswith("Unnamed")])
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert("UTC").tz_localize(None)
    if "community" not in prices:
        prices["community"] = 0.2

    solar = None
    if args.pv:
        pv = pd.read_parquet(args.pv)
        print(f"pv columns: {list(pv.columns)[:8]} ... take the one that fits and edit solar.csv by hand if needed")
        solar = pv.select_dtypes("number")
        if "datetime" in pv.columns:
            solar.index = pd.to_datetime(pv["datetime"])
        if getattr(solar.index, "tz", None) is not None:
            solar.index = solar.index.tz_convert("UTC").tz_localize(None)
        solar = solar.iloc[:, [0]].rename(columns=lambda c: "pv_0").reindex(loads.index, method="nearest")

    rows = []
    for k, pid in enumerate(profiles):
        seller = k >= len(profiles) - args.sellers
        rows.append(
            {
                "unit_id": f"{'S' if seller else 'B'}_{k + 1:02d}",
                "role": "sell" if seller else "buy",
                "location": args.location,
                "load_profile": f"load_{pid}",
                "solar_profile": "pv_0" if solar is not None else "",
                "battery_kwh": 10 if seller else 0,
                "tradeable_pct": 80,
                "risk_tolerance": "Low",
                "trading_scope": "All",
            }
        )
    write_inputs(Path(args.out), pd.DataFrame(rows), loads.round(4), prices, solar)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    syn = sub.add_parser("synthetic")
    syn.add_argument("--out", required=True)
    syn.add_argument("--start", default="2023-05-01")
    syn.add_argument("--days", type=float, default=42)
    syn.add_argument("--buyers", type=int, default=3)
    syn.add_argument("--sellers", type=int, default=2)
    syn.add_argument("--seed", type=int, default=1)
    syn.set_defaults(func=synthetic)

    syl = sub.add_parser("synload")
    syl.add_argument("--out", required=True)
    syl.add_argument("--timeseries", required=True, help="synload-de timeseries.parquet")
    syl.add_argument("--prices", required=True, help="prices.csv with wholesale, eeg, grid, community")
    syl.add_argument("--pv", default=None, help="PV profile (parquet), optional")
    syl.add_argument("--profiles", nargs="+", type=int, required=True, help="profile_ids from synload-de")
    syl.add_argument("--sellers", type=int, default=1, help="the last N profiles become sellers")
    syl.add_argument("--location", default="aachen")
    syl.set_defaults(func=synload)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
