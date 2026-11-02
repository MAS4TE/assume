# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Input and output of the grid simulation.

Inputs  (``load_inputs``): a folder with ``units.csv``, ``loads.csv``,
        ``prices.csv`` and optionally ``solar.csv``, see GRID_SIM.md.
Outputs (``Recorder``):    bids, trades, per-household flows and the physical
        battery schedules of the sellers, written to one folder per run.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PRICE_COLUMNS = ("supplier", "eeg", "wholesale", "community")
ROLES = {"buy": "buy", "buyer": "buy", "sell": "sell", "seller": "sell"}
EPS = 1e-9


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@dataclass
class UnitSpec:
    """One household: a buyer (rents storage) or a seller (owns a battery)."""

    unit_id: str
    role: str  # "buy" or "sell"
    location: str
    load_kw: pd.Series
    solar_kw: pd.Series
    battery_kwh: float = 0.0
    tradeable_pct: float = 50.0
    risk_tolerance: str = "Low"
    trading_scope: str = "All"


@dataclass
class Inputs:
    units: list[UnitSpec]
    prices: dict[str, pd.Series]
    step_hours: float
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive, end of the data
    locations: list[str] = field(default_factory=list)


def _read_wide(path: Path) -> pd.DataFrame:
    """Read a ``datetime`` indexed table, timezone-naive, sorted."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if "datetime" in df.columns:
            df = df.set_index("datetime")
    else:
        df = pd.read_csv(path, comment="#")
        time_col = "datetime" if "datetime" in df.columns else df.columns[0]
        df = df.set_index(time_col)
    index = pd.to_datetime(df.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    df.index = index
    df.index.name = "datetime"
    return df.sort_index()


def _find(folder: Path, stem: str) -> Path | None:
    for suffix in (".csv", ".parquet"):
        if (folder / f"{stem}{suffix}").exists():
            return folder / f"{stem}{suffix}"
    return None


def load_inputs(folder: str | Path) -> Inputs:
    """Load and check the input folder, see GRID_SIM.md for the format."""
    folder = Path(folder)
    units_path = _find(folder, "units")
    loads_path = _find(folder, "loads")
    prices_path = _find(folder, "prices")
    if units_path is None or loads_path is None or prices_path is None:
        raise FileNotFoundError(
            f"{folder} needs units.csv, loads.csv and prices.csv (solar.csv is optional)"
        )

    units_df = pd.read_csv(units_path, dtype={"unit_id": str}, comment="#")
    loads = _read_wide(loads_path)
    solar_path = _find(folder, "solar")
    solar = _read_wide(solar_path) if solar_path else pd.DataFrame(index=loads.index)

    # step size from the load profiles, everything else is aligned to it
    step = loads.index.to_series().diff().dropna().mode().iloc[0]
    step_hours = step.total_seconds() / 3600
    prices_raw = _read_wide(prices_path).rename(columns={"grid": "supplier"})
    missing = [c for c in PRICE_COLUMNS if c not in prices_raw.columns]
    if missing:
        raise ValueError(f"prices.csv misses the columns {missing}")
    prices = {
        c: prices_raw[c].astype(float).reindex(loads.index, method="nearest")
        for c in PRICE_COLUMNS
    }

    units = []
    for row in units_df.itertuples(index=False):
        role = ROLES.get(str(row.role).strip().lower())
        if role is None:
            raise ValueError(f"unit {row.unit_id}: role must be buy or sell")
        if row.load_profile not in loads.columns:
            raise ValueError(f"unit {row.unit_id}: load_profile {row.load_profile!r} not in loads")

        solar_name = getattr(row, "solar_profile", None)
        if isinstance(solar_name, str) and solar_name.strip():
            if solar_name not in solar.columns:
                raise ValueError(f"unit {row.unit_id}: solar_profile {solar_name!r} not in solar")
            solar_kw = solar[solar_name].astype(float).reindex(loads.index, method="nearest")
        else:
            solar_kw = pd.Series(0.0, index=loads.index)

        def opt(name, default):
            value = getattr(row, name, default)
            return default if pd.isna(value) else value

        units.append(
            UnitSpec(
                unit_id=str(row.unit_id),
                role=role,
                location=str(opt("location", "aachen")).lower(),
                load_kw=loads[row.load_profile].astype(float),
                solar_kw=solar_kw,
                battery_kwh=float(opt("battery_kwh", 0.0)),
                tradeable_pct=float(opt("tradeable_pct", 50.0)),
                risk_tolerance=str(opt("risk_tolerance", "Low")),
                trading_scope=str(opt("trading_scope", "All")),
            )
        )
        if role == "sell" and units[-1].battery_kwh <= 0:
            raise ValueError(f"seller {row.unit_id} needs battery_kwh > 0")

    if len({u.unit_id for u in units}) != len(units):
        raise ValueError("unit_id must be unique")

    return Inputs(
        units=units,
        prices=prices,
        step_hours=step_hours,
        start=loads.index[0],
        end=loads.index[-1] + step,
        locations=sorted({u.location for u in units}),
    )


def window(series: pd.Series, start, end) -> pd.Series:
    """The part of ``series`` in [start, end), error if the data does not cover it."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    part = series[(series.index >= start) & (series.index < end)]
    step = series.index.to_series().diff().dropna().mode().iloc[0]
    expected = int(round((end - start) / step))
    if len(part) != expected:
        raise ValueError(
            f"input data covers {len(part)} of {expected} steps for {start} to {end}, "
            "extend the input files or shorten the simulation"
        )
    return part


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
class Recorder:
    """Collects bids, results and schedules during the run and writes them at the end.

    One instance is shared by all strategies. Files in ``out_dir``:

    - ``bids.csv``              every submitted bid with the result of the clearing
    - ``trades.csv``            every match: buyer, seller, volume, price, trade_id
    - ``household_flows.csv``   energy flows of the households (only awarded ones)
    - ``household_summary.csv`` per household and week: volumes, rent, cashflows
    - ``battery_schedule.csv``  physical battery of the sellers (own use + rented out)
    """

    def __init__(self, out_dir: str | Path, fmt: str = "csv", export_unawarded: bool = False):
        self.out_dir = Path(out_dir)
        self.fmt = fmt
        self.export_unawarded = export_unawarded
        self.bid_unit: dict[str, str] = {}
        self.bids: list[dict] = []
        self.trades: list[dict] = []
        self.weeks: list[dict] = []  # one entry per unit and week, incl. the flows frame

    # -- while running -----------------------------------------------------
    def register_bids(self, unit_id: str, bid_ids: list[str]) -> None:
        for bid_id in bid_ids:
            self.bid_unit[bid_id] = unit_id

    def record_result(self, unit_id: str, role: str, orders: list[dict]) -> None:
        for order in orders:
            volume = abs(order.get("accepted_volume") or 0.0)
            self.bids.append(
                {
                    "unit_id": unit_id,
                    "role": role,
                    "bid_id": order["bid_id"],
                    "product_start": order["start_time"],
                    "product_end": order["end_time"],
                    "location": order.get("location"),
                    "exclusive_id": order.get("exclusive_id"),
                    "bid_volume_kwh": abs(order["volume"]),
                    "bid_price_eur_kwh": order["price"],
                    "accepted_volume_kwh": volume,
                    "accepted_price_eur_kwh": order.get("accepted_price") or 0.0,
                }
            )
            if role != "buy" or volume < EPS:
                continue
            matches = order.get("matches") or []
            if isinstance(matches, str):
                matches = json.loads(matches)
            for match in matches:
                self.trades.append(
                    {
                        "trade_id": match["trade_id"],
                        "product_start": order["start_time"],
                        "product_end": order["end_time"],
                        "location": order.get("location"),
                        "buyer_unit": unit_id,
                        "seller_unit": self.bid_unit.get(match["bid_id"]),
                        "buyer_bid_id": order["bid_id"],
                        "seller_bid_id": match["bid_id"],
                        "volume_kwh": float(match["volume"]),
                        "price_eur_kwh": float(match["price"]),
                    }
                )

    def record_week(self, entry: dict) -> None:
        self.weeks.append(entry)

    # -- at the end --------------------------------------------------------
    def _write(self, df: pd.DataFrame, name: str, index: bool = False) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.fmt == "parquet":
            df.to_parquet(self.out_dir / f"{name}.parquet", index=index)
        else:
            df.to_csv(self.out_dir / f"{name}.csv", index=index)

    def _battery_schedule(self, flows: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """Physical battery power of each seller.

        A buyer's schedule describes the rented *virtual* storage. It is split
        over the sellers it rented from in proportion to the matched volumes
        and added to the sellers' own-use schedule.
        """
        rows = []
        buyers = flows[flows["role"] == "buy"]
        sellers = flows[flows["role"] == "sell"]

        for week_start, week in sellers.groupby("product_start"):
            for unit_id, own in week.groupby("unit_id"):
                frame = own.set_index("timestamp")[["storage_charge_kw"]].rename(
                    columns={"storage_charge_kw": "own_use_kw"}
                )
                frame["rented_out_kw"] = 0.0
                rented = trades[
                    (trades["seller_unit"] == unit_id)
                    & (trades["product_start"] == week_start)
                ]
                for buyer_id, part in rented.groupby("buyer_unit"):
                    total = trades[
                        (trades["buyer_unit"] == buyer_id)
                        & (trades["product_start"] == week_start)
                    ]["volume_kwh"].sum()
                    tenant = buyers[
                        (buyers["unit_id"] == buyer_id)
                        & (buyers["product_start"] == week_start)
                    ].set_index("timestamp")["storage_charge_kw"]
                    share = part["volume_kwh"].sum() / total
                    frame["rented_out_kw"] = frame["rented_out_kw"].add(
                        tenant * share, fill_value=0.0
                    )
                frame["total_kw"] = frame["own_use_kw"] + frame["rented_out_kw"]
                frame.insert(0, "unit_id", unit_id)
                frame.insert(1, "product_start", week_start)
                rows.append(frame.reset_index())
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def write(self) -> None:
        """Write all result files, also callable after an aborted run."""
        self._write(pd.DataFrame(self.bids), "bids")
        trades = pd.DataFrame(self.trades)
        self._write(trades, "trades")

        if not self.weeks:
            log.warning("no household schedules recorded, nothing was awarded")
            return

        flow_frames, summary = [], []
        for week in self.weeks:
            flows = week["flows"].copy()
            flows.insert(0, "timestamp", flows.index)
            for name in ("unit_id", "role", "location", "product_start"):
                flows.insert(1, name, week[name])
            flow_frames.append(flows.reset_index(drop=True))
            summary.append({k: v for k, v in week.items() if k != "flows"})
        flows = pd.concat(flow_frames, ignore_index=True)
        # reorder the identifying columns
        lead = ["timestamp", "unit_id", "role", "location", "product_start"]
        flows = flows[lead + [c for c in flows.columns if c not in lead]]

        self._write(flows, "household_flows")
        self._write(pd.DataFrame(summary), "household_summary")
        battery = self._battery_schedule(flows, trades if len(trades) else pd.DataFrame(
            columns=["seller_unit", "buyer_unit", "product_start", "volume_kwh"]
        ))
        if len(battery):
            self._write(battery, "battery_schedule")
        log.info("results written to %s", self.out_dir)
