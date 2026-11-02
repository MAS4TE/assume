# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Weekly battery market with BUC bidding, for the grid simulation.

Households (buyers and sellers) come from an input folder, they bid with the
Battery Utility Calculator in-process (no MQTT, no agents), the market clears
weekly and the schedules of the awarded households are written to a result
folder. See GRID_SIM.md.

    python simulation.py --inputs inputs/example --out results/example
"""

import argparse
import logging
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
from battery_market_clearing import BatteryClearing
from buc_strategy import BUCSettings, BUCStrategy
from dateutil import rrule as rr
from grid_sim_io import Inputs, Recorder, load_inputs
from battery_utility_calculator import Storage

try:  # ElectricVehicle is only available in newer BUC versions
    from battery_utility_calculator import ElectricVehicle
except ImportError:
    ElectricVehicle = None

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

log = logging.getLogger(__name__)

MARKET_ID = "BatteryMarket"


# --- EV helpers ---------------------------------------------------------------
# Not used by the grid simulation yet (the EV load of synload-de is part of the
# fixed household load). Kept from the MQTT simulation, tests/test_mas4te_ev.py uses them.
# Static parameters per vehicle type (types follow synload-de: tesla, vw_eup, none).
# capacity_kwh = usable battery capacity, max_charge_kw = AC charging power at the
# home wallbox, max_discharge_kw > 0 would enable V2H (car -> home only).
EV_TYPES = {
    "tesla": {"capacity_kwh": 57.5, "max_charge_kw": 11.0, "max_discharge_kw": 0.0},
    "vw_eup": {"capacity_kwh": 32.3, "max_charge_kw": 7.2, "max_discharge_kw": 0.0},
}


def read_ev_profile(index, id: str = "0", data_dir: str = "./example_data"):
    """Reads the EV time series of profile ``id`` from ``ev.csv`` and aligns them to ``index``.

    ``ev.csv`` has one pair of columns per profile: ``ev_availability_<id>``
    (1 = at home and plugged in, 0 = away) and ``ev_driving_consumption_kw_<id>``
    (kW drawn from the battery while driving). A file with the unsuffixed columns
    ``ev_availability`` / ``ev_driving_consumption_kw`` (one vehicle per file, see
    the EV data contract) is accepted as well.

    If the file covers the whole ``index`` it is used as is, otherwise (e.g. the
    one-week placeholder profile) it is repeated by weekday and time of day.

    Args:
        index (FastIndex | pd.DatetimeIndex): The index of the simulation.
        id (str, optional): The profile ID. Defaults to "0".
        data_dir (str, optional): Folder containing ``ev.csv``.

    Returns:
        dict: ``ev_availability`` and ``ev_driving_consumption_kw`` as pd.Series on ``index``.
    """
    if hasattr(index, "as_datetimeindex"):
        index = index.as_datetimeindex()

    profiles = pd.read_csv(
        f"{data_dir}/ev.csv", index_col=0, parse_dates=True, comment="#"
    )
    result = {}
    for name in ("ev_availability", "ev_driving_consumption_kw"):
        column = f"{name}_{id}" if f"{name}_{id}" in profiles else name
        profile = profiles[column]
        if profile.index[0] <= index[0] and profile.index[-1] >= index[-1]:
            series = profile.reindex(index)
        else:
            # repeat the profile week by weekday and time of day
            week = profile.groupby(
                [profile.index.day_of_week, profile.index.hour, profile.index.minute]
            ).first()
            keys = list(zip(index.day_of_week, index.hour, index.minute))
            series = pd.Series(week.reindex(keys).to_numpy(), index=index)
        if series.isna().any():
            raise ValueError(f"EV profile {column} does not cover the simulation index")
        result[name] = series.astype(float).rename(name)

    return result


def create_electric_vehicle(id, ev_type: str, ev_profile: dict, **kwargs):
    """Creates the BUC ElectricVehicle of a household from its type and profile.

    Args:
        id: The ID of the vehicle.
        ev_type (str): Key of ``EV_TYPES``.
        ev_profile (dict): Output of ``read_ev_profile``.
        **kwargs: Further ElectricVehicle parameters (e.g. ``min_departure_soc``).

    Returns:
        ElectricVehicle: The vehicle, to be passed as ``electric_vehicle`` to the unit.
    """
    if ElectricVehicle is None:
        raise ImportError(
            "battery_utility_calculator has no ElectricVehicle, "
            "install a BUC version with EV support or set n_ev_households = 0"
        )
    return ElectricVehicle(
        id=id,
        availability=ev_profile["ev_availability"],
        driving_consumption=ev_profile["ev_driving_consumption_kw"],
        **EV_TYPES[ev_type],
        **kwargs,
    )
# ------------------------------------------------------------------------------


def build_market(start, end) -> MarketConfig:
    """Weekly storage market: each product is the rent of one week, delivered 12 h after the close."""
    return MarketConfig(
        market_id=MARKET_ID,
        opening_hours=rr.rrule(rr.WEEKLY, interval=1, dtstart=start, until=end, cache=True),
        opening_duration=timedelta(hours=1),
        market_mechanism="battery_clearing",
        product_type="power",
        market_products=[
            MarketProduct(
                duration=timedelta(days=7),
                count=1,  # only the next week is traded
                first_delivery=timedelta(hours=12),
            )
        ],
        additional_fields=["c_rate", "blockchain_address", "location", "exclusive_id"],
        param_dict={"allowed_c_rates": [1]},
        minimum_bid_price=0,
        volume_unit="kW",
    )


def init(
    world: World,
    inputs: Inputs,
    recorder: Recorder,
    settings: BUCSettings,
    weeks: int,
    start=None,
    simulation_id: str = "mas4te_grid_sim",
) -> None:
    """Set up the world: one weekly market and one unit per household of ``inputs``."""
    start = pd.Timestamp(start if start is not None else inputs.start).to_pydatetime()
    # ASSUME opens a market only if its product ends before ``until`` of the opening
    # rule, so the world runs until the delivery of the last of the ``weeks`` markets ends
    delivery_end = start + timedelta(weeks=weeks, hours=12)
    end = delivery_end + timedelta(hours=1)
    if delivery_end > inputs.end.to_pydatetime():
        raise ValueError(
            f"{weeks} weeks from {start} need data until {delivery_end}, "
            f"the inputs end at {inputs.end}. Reduce --weeks or extend the input files."
        )

    # ASSUME needs a regular index for its own bookkeeping, the BUC reads the
    # profiles in their own resolution from the inputs
    index = FastIndex(start, end + timedelta(weeks=2), freq="h")

    world.bidding_strategies["buc_strategy"] = BUCStrategy
    world.clearing_mechanisms["battery_clearing"] = BatteryClearing
    world.setup(start=start, end=end, save_frequency_hours=48, simulation_id=simulation_id)

    market_config = build_market(start, end)
    world.add_market_operator(id="market_operator")
    world.add_market("market_operator", market_config)

    for spec in inputs.units:
        world.add_unit_operator(id=f"operator_{spec.unit_id}")
        world.add_unit(
            id=spec.unit_id,
            unit_type="mas4te",
            unit_operator_id=f"operator_{spec.unit_id}",
            unit_params={
                "baseline_storage": Storage(
                    id=0,
                    c_rate=settings.c_rate,
                    volume=spec.battery_kwh if spec.role == "sell" else 0,
                ),
                "max_power": 1000,
                "min_power": 0,
                "bidding_strategies": {MARKET_ID: "buc_strategy"},
                "bidding_params": {
                    "spec": spec,
                    "inputs": inputs,
                    "recorder": recorder,
                    "settings": settings,
                    "market_config": market_config,
                },
                "technology": "battery_storage" if spec.role == "sell" else "demand",
            },
            # ASSUME requires a forecaster, the profiles themselves are held by the strategy
            forecaster=NaiveForecast(index=index, demand=0, availability=1),
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="input folder (units.csv, loads.csv, prices.csv, ...)")
    parser.add_argument("--out", required=True, help="result folder, created if missing")
    parser.add_argument("--weeks", type=int, default=4, help="number of weekly markets (default 4)")
    parser.add_argument("--start", default=None, help="first market opening, e.g. 2023-05-08T13:00 (default: start of the inputs)")
    parser.add_argument("--solver", default="appsi_highs", help="pyomo solver of the BUC (default appsi_highs)")
    parser.add_argument("--risk", action="store_true", help="risk adjusted bidding curve (scenarios) instead of the deterministic one")
    parser.add_argument("--scenarios", type=int, default=12, help="scenarios of the risk adjusted bidding (with --risk)")
    parser.add_argument("--seed", type=int, default=42, help="seed of the scenarios, -1 for a different one in every run")
    parser.add_argument("--goal", default="max_cashflow", choices=["max_cashflow", "max_green_energy"])
    parser.add_argument("--format", default="csv", choices=["csv", "parquet"], help="format of the result files")
    parser.add_argument("--export-unawarded", action="store_true",
                        help="also write flows of households that were not awarded (default: only awarded)")
    parser.add_argument("--db", default=None, help="ASSUME database uri (default: sqlite file in the result folder)")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)
    # the results of this simulation must not reach the BRP, even if a .env is around
    os.environ.pop("BRP_MARKET_RESULTS_URL", None)

    inputs = load_inputs(args.inputs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(out, fmt=args.format, export_unawarded=args.export_unawarded)
    settings = BUCSettings(
        solver=args.solver,
        risk_bidding=args.risk,
        n_scenarios=args.scenarios,
        seed=None if args.seed < 0 else args.seed,
        goal=args.goal,
    )

    world = World(database_uri=args.db or f"sqlite:///{out / 'assume.db'}", log_level="ERROR")
    start = pd.Timestamp(args.start) if args.start else inputs.start
    init(world, inputs, recorder, settings, weeks=args.weeks, start=start)
    try:
        world.run()
    finally:
        # also after an aborted run, the weeks that are done are kept
        recorder.write()


if __name__ == "__main__":
    main()
