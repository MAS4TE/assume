# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg2
from battery_utility_calculator import Storage
from dateutil import rrule as rr
from mas4te_bidding_strategy import LLMBuyStrategy, LLMSellStrategy
from mas4te_clearing_mechanism import BatteryClearing

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

np.random.seed(seed=42)
log = logging.getLogger(__name__)


def read_forecasts():
    """Reads the forecasts for a specific time period and unit ID.

    Args:
        start (datetime): The start time of the forecast period.
        end (datetime): The end time of the forecast period.
        id (int, optional): The ID of the unit to read forecasts for. Defaults to 0.
        randomize (bool, optional): Whether to randomize the forecasts. Defaults to False. Overwrites the ID.

    Returns:
        dict: A dictionary containing the forecasts for the specified time period and unit ID.
    """
    demand_forecast = pd.read_csv(
        "./analysis_data/profile_timeseries.csv", index_col=0
    ).set_index("datetime")
    prices = pd.read_csv("./analysis_data/prices.csv", index_col=0).set_index(
        "datetime"
    )
    solar_gen = pd.read_csv("./analysis_data/solar.csv", index_col=0).set_index(
        "datetime"
    )

    return {
        "demand": demand_forecast,
        "prices": prices,
        "solar_gen": solar_gen,
    }


def init(
    world: World, db_uri: str, sim_id: str, n_supply_units: int, n_demand_units: int
):
    con = psycopg2.connect(db_uri)

    # set start and end date
    start = datetime(2023, 1, 1, hour=13)
    end = datetime(2023, 12, 31, hour=23)

    # create index
    index = FastIndex(start, end, freq="h")

    # set simulation ID
    simulation_id = f"{sim_id}"

    try:
        existing_ids = pd.read_sql("SELECT simulation FROM sim_config", db_uri)[
            "simulation"
        ].values
    except Exception:
        existing_ids = []

    if simulation_id in existing_ids:
        msg = f"Simulation with ID {simulation_id} already exists! Overwrite (Y/n)? "
        user_input = input(msg)
        if user_input == "y" or user_input == "":
            pass
        else:
            return

    # add possible bidding strategies
    world.bidding_strategies["llm_buy_strategy"] = LLMBuyStrategy
    world.bidding_strategies["llm_sell_strategy"] = LLMSellStrategy

    # add possible clearing mechanism
    world.clearing_mechanisms["battery_clearing"] = BatteryClearing

    # set up world
    world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=simulation_id,
    )

    # create market design
    market_id = "BatteryMarket"
    market_products = [
        MarketProduct(
            id=0,
            duration=timedelta(hours=24 * 7),
            count=1,
            first_delivery=timedelta(hours=12),
        )
    ]
    for mp in market_products:
        try:
            existing_ids = pd.read_sql("SELECT id FROM market_products", db_uri)[
                "id"
            ].values
        except Exception:
            existing_ids = []

        # if mp.id in existing_ids:
        #     msg = f"Market Product with ID {mp.id} already exists! Overwrite (Y/n)? "
        #     user_input = input(msg)
        #     if user_input == "y" or user_input == "":
        #         pass
        #     else:
        #         return

    marketdesign = [
        MarketConfig(
            market_id=market_id,
            opening_hours=rr.rrule(
                rr.WEEKLY,
                interval=1,
                dtstart=start,
                until=end,
                cache=True,  # weekly battery market with the next week tradeable
            ),
            opening_duration=timedelta(hours=1),
            market_mechanism="battery_clearing",
            product_type="power",
            market_products=market_products,
            additional_fields=["c_rate"],
            param_dict={"allowed_c_rates": [1]},
            minimum_bid_price=0,
            volume_unit="kW",
        )
    ]

    # create and add market operator
    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)

    # add market to world
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    forecasts = read_forecasts()
    log.info("Read timeseries")

    ##################################################
    # SET THE NUMBER OF DEMAND AND SUPPLY UNITS HERE #
    ##################################################
    n_supply_units = n_supply_units
    n_demand_units = n_demand_units

    if int(n_supply_units) > 50:
        supply_profiles = list(range(int(n_supply_units)))
    else:
        supply_profiles = np.random.randint(low=0, high=119, size=int(n_supply_units))

    if int(n_demand_units) > 50:
        demand_profiles = list(range(int(n_demand_units)))
    else:
        demand_profiles = np.random.randint(low=0, high=119, size=int(n_demand_units))

    # redo if we have duplicate IDs
    while len(supply_profiles) != len(set(supply_profiles)):
        supply_profiles = np.random.randint(low=0, high=119, size=int(n_supply_units))
    while len(demand_profiles) != len(set(demand_profiles)):
        demand_profiles = np.random.randint(low=0, high=119, size=int(n_demand_units))

    # actually create and add the demand units
    for demand_id in demand_profiles:
        world.add_unit_operator(id=f"storage_demand_operator_{demand_id}")
        world.add_unit(
            id=f"storage_demand_{demand_id}",
            unit_type="mas4te",
            unit_operator_id=f"storage_demand_operator_{demand_id}",
            unit_params={
                "baseline_storage": Storage(id=0, c_rate=1, volume=0, efficiency=1),
                "max_power": 1000,
                "min_power": 0,
                "bidding_strategies": {"BatteryMarket": "llm_buy_strategy"},
                "technology": "demand",
            },
            forecaster=NaiveForecast(
                index=index,
                demand=0,
                energy_demand=forecasts["demand"].query(f"profile_id == {demand_id}")[
                    "load_kw"
                ],
                wholesale_price=forecasts["prices"]["wholesale"],
                eeg_price=forecasts["prices"]["eeg"],
                community_price=forecasts["prices"]["community"],
                grid_price=forecasts["prices"]["grid"],
                solar_gen=forecasts["solar_gen"].query(f"profile_id == {demand_id}")[
                    "solar_gen_kw"
                ],
            ),
        )

    # actually create and add the supply units
    for supply_id in supply_profiles:
        storage_volume = np.random.normal(loc=8.5422, scale=3.155)
        storage_volume = 0 if storage_volume < 0 else storage_volume

        world.add_unit_operator(f"storage_supply_operator_{supply_id}")
        world.add_unit(
            id=f"storage_supply_{supply_id}",
            unit_type="mas4te",
            unit_operator_id=f"storage_supply_operator_{supply_id}",
            unit_params={
                "baseline_storage": Storage(
                    id=0, c_rate=1, volume=storage_volume, efficiency=0.95
                ),
                "max_power_charge": 1,
                "max_power_discharge": 1,
                "max_soc": 20,
                "min_soc": 0,
                "efficiency_charge": 0.975,
                "efficiency_discharge": 0.975,
                "bidding_strategies": {"BatteryMarket": "llm_sell_strategy"},
                "technology": "battery_storage",
            },
            forecaster=NaiveForecast(
                index=index,
                demand=0,
                energy_demand=forecasts["demand"].query(f"profile_id == {supply_id}")[
                    "load_kw"
                ],
                wholesale_price=forecasts["prices"]["wholesale"],
                eeg_price=forecasts["prices"]["eeg"],
                community_price=forecasts["prices"]["community"],
                grid_price=forecasts["prices"]["grid"],
                solar_gen=forecasts["solar_gen"].query(f"profile_id == {supply_id}")[
                    "solar_gen_kw"
                ],
            ),
        )

    sim_config = pd.DataFrame(
        data={
            "simulation": simulation_id,
            "start": start,
            "end": end,
            "n_supply_units": n_supply_units,
            "n_demand_units": n_demand_units,
            "market_id": market_id,
            "product_ids": [[prod.id for prod in market_products]],
        },
        index=[0],
    )
    try:
        cur = con.cursor()
        cur.execute(f"DELETE FROM sim_config WHERE simulation = '{simulation_id}'")
        con.commit()
    except psycopg2.errors.UndefinedTable:
        con.rollback()
    sim_config.to_sql("sim_config", db_uri, if_exists="append")

    for prod in market_products:
        df = pd.DataFrame(
            data={
                "id": prod.id,
                "duration_d": prod.duration.days,
                "count": prod.count,
                "first_delivery_h": prod.first_delivery.seconds / 3600,
            },
            index=[0],
        )
        try:
            cur = con.cursor()
            cur.execute(f"DELETE FROM market_products WHERE id = {prod.id}")
            con.commit()
        except psycopg2.errors.UndefinedTable:
            con.rollback()
        df.to_sql("market_products", db_uri, if_exists="append")


def main():
    logging.getLogger("gurobipy").setLevel(logging.WARNING)  # suppress gurobipy logs

    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument("--sim-id")
    parser.add_argument("--n-supply")
    parser.add_argument("--n-demand")

    args = parser.parse_args()

    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(
        database_uri=db_uri,
        log_level="ERROR",
    )
    init(
        world,
        db_uri,
        sim_id=args.sim_id,
        n_supply_units=args.n_supply,
        n_demand_units=args.n_demand,
    )

    start = datetime.now().replace(microsecond=0)
    world.run()
    end = datetime.now().replace(microsecond=0)
    msg = (
        f"Started on {start.isoformat()}, ended on {end.isoformat()}, took {end-start}"
    )
    print(msg)


if __name__ == "__main__":
    main()
