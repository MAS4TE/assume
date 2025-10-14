# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import random
from datetime import datetime, timedelta

import pandas as pd
from battery_utility_calculator import Storage
from dateutil import rrule as rr
from mas4te_bidding_strategy import LLMBuyStrategy, LLMSellStrategy
from mas4te_clearing_mechanism import BatteryClearing

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

log = logging.getLogger(__name__)


def read_forecasts(start, end, id: int = 0, randomize: bool = False):
    """Reads the forecasts for a specific time period and unit ID.

    Args:
        start (datetime): The start time of the forecast period.
        end (datetime): The end time of the forecast period.
        id (int, optional): The ID of the unit to read forecasts for. Defaults to 0.
        randomize (bool, optional): Whether to randomize the forecasts. Defaults to False. Overwrites the ID.

    Returns:
        dict: A dictionary containing the forecasts for the specified time period and unit ID.
    """
    if randomize:
        val = random.randint(0, 29)
        id = "0" + str(val) if val < 10 else str(val)

    demand_forecast = pd.read_csv(
        "./example_data/demand.csv", index_col=0, parse_dates=True
    )["demand" + "_" + id][start:end]
    wholesale_price = pd.read_csv(
        "./example_data/prices.csv", index_col=0, parse_dates=True
    )["wholesale"][start:end]
    eeg_price = pd.read_csv("./example_data/prices.csv", index_col=0, parse_dates=True)[
        "eeg"
    ][start:end]
    community_price = pd.read_csv(
        "./example_data/prices.csv", index_col=0, parse_dates=True
    )["community"][start:end]
    grid_price = pd.read_csv(
        "./example_data/prices.csv", index_col=0, parse_dates=True
    )["grid"][start:end]
    solar_gen = pd.read_csv("./example_data/solar.csv", index_col=0, parse_dates=True)[
        "solar" + "_" + id
    ][start:end]

    return {
        "demand": demand_forecast,
        "wholesale_price": wholesale_price,
        "eeg_price": eeg_price,
        "community_price": community_price,
        "grid_price": grid_price,
        "solar_gen": solar_gen,
    }


def init(world: World, n=1):
    # set start and end date
    start = datetime(2023, 1, 1, hour=13)
    end = datetime(2023, 1, 8, hour=13)

    # create index
    index = FastIndex(start, end, freq="h")

    # set simulation ID
    simulation_id = "mas4te_simulation"

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
    marketdesign = [
        MarketConfig(
            market_id="BatteryMarket",
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
            market_products=[
                MarketProduct(
                    duration=timedelta(
                        hours=24 * 7
                    ),  # each product (storage rent) will be 1 week
                    count=1,  # we will only trade the next week, not any week after that
                    first_delivery=timedelta(hours=12),
                )
            ],  # delivery will take place 12 hours after market close
            additional_fields=["c_rate"],
            param_dict={"allowed_c_rates": [1]},
            minimum_bid_price=0,
        )
    ]

    # create and add market operator
    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)

    # add market to world
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    ##################################################
    # SET THE NUMBER OF DEMAND AND SUPPLY UNITS HERE #
    ##################################################
    n_demand_units = 1
    n_supply_units = 1

    # actually create and add the demand units
    for i in range(n_demand_units):
        # we need a demand, solar generation and price forecast to build bids
        # so we have to read them in before providing them to the forecaster of the unit
        # -----------------------------------------------------------------------------------
        # you can provide an ID (0 to 29) here and the forecast for that ID will be read in
        # or you can set "randomize" to True, to choose a random forecast
        forecasts = read_forecasts(start, end, id=str(i), randomize=False)

        id = "0" + str(i + 1) if i < 9 else str(i + 1)
        world.add_unit_operator(id=f"storage_demand_operator_{id}")
        world.add_unit(
            id=f"storage_demand_{id}",
            unit_type="mas4te",
            unit_operator_id=f"storage_demand_operator_{id}",
            unit_params={
                "baseline_storage": Storage(
                    id=0, c_rate=1, volume=0, efficiency=1
                ),  # unit has no storage
                "max_power": 1000,  # max 1.000 kW demand
                "min_power": 0,  # no minimum demand
                "bidding_strategies": {"BatteryMarket": "llm_buy_strategy"},
                "technology": "demand",
            },
            forecaster=NaiveForecast(
                index=index,
                demand=0,
                energy_demand=forecasts["demand"],
                wholesale_price=forecasts["wholesale_price"],
                eeg_price=forecasts["eeg_price"],
                community_price=forecasts["community_price"],
                grid_price=forecasts["grid_price"],
                solar_gen=forecasts["solar_gen"],
            ),
        )

    # actually create and add the supply units
    for i in range(n_supply_units):
        # same as above - set an ID or set randomize to True
        forecasts = read_forecasts(start, end, id=str(i))

        id = "0" + str(i + 1) if i < 9 else str(i + 1)
        world.add_unit_operator(f"storage_provider_operator_{id}")
        world.add_unit(
            id=f"storage_provider_{id}",
            unit_type="mas4te",
            unit_operator_id=f"storage_provider_operator_{id}",
            unit_params={
                "baseline_storage": Storage(id=0, c_rate=1, volume=20, efficiency=0.95),
                "max_power_charge": 1,  # max 1 kW charge
                "max_power_discharge": 1,  # max 1 kW discharge
                "max_soc": 20,  # max 20 kWh of storage capacity (equal to baseline)
                "min_soc": 0,  # no mimimum fill level
                "efficiency_charge": 0.975,  # charge and discharge to combine to 95% efficiency
                "efficiency_discharge": 0.975,
                "bidding_strategies": {"BatteryMarket": "llm_sell_strategy"},
                "technology": "battery_storage",
            },
            forecaster=NaiveForecast(
                index=index,
                demand=0,
                availability=1,  # always available
                energy_demand=forecasts["demand"],
                solar_gen=forecasts["solar_gen"],
                wholesale_price=forecasts["wholesale_price"],
                eeg_price=forecasts["eeg_price"],
                community_price=forecasts["community_price"],
                grid_price=forecasts["grid_price"],
                # no battery demand, fuel price or CO2 price for this simulation
                battery_demand=0,  # no battery demand
                fuel_price=0,  # no fuel price
                co2_price=0,  # no CO2 price
            ),
        )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri, log_level="ERROR")
    init(world)
    logging.getLogger("gurobipy").setLevel(logging.WARNING)  # suppress gurobipy logs
    world.run()
