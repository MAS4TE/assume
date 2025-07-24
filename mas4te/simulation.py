# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta

# from mas4te_clearing_mechanism import BatteryClearing
import pandas as pd
from dateutil import rrule as rr
from mas4te_bidding_strategy import LLMBuyStrategy, LLMSellStrategy

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

log = logging.getLogger(__name__)


def read_forecasts(start, end):
    demand_forecast = pd.read_csv(
        "./example_data/demand.csv", index_col=0, parse_dates=True
    )["demand"][start:end]
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
    solar_generation_forecast = pd.read_csv(
        "./example_data/solar.csv", index_col=0, parse_dates=True
    )["solar"][start:end]

    return {
        "demand": demand_forecast,
        "wholesale_price": wholesale_price,
        "eeg_price": eeg_price,
        "community_price": community_price,
        "grid_price": grid_price,
        "solar_generation_forecast": solar_generation_forecast,
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
    # TODO implement BatteryClearing
    # world.clearing_mechanisms["battery_clearing"] = BatteryClearing

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
            market_mechanism="pay_as_clear",
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
            additional_fields=["c_rate", "efficiency"]
        )
    ]

    # create and add market operator
    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)

    # add market to world
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    # create and add demand (buy) unit
    # we need a demand, solar generation and price forecast to build bids
    # so we have to read them in before providing them to the forecaster of the unit
    forecasts = read_forecasts(start, end)

    ##################################################
    # SET THE NUMBER OF DEMAND AND SUPPLY UNITS HERE #
    ##################################################
    n_demand_units = 1
    n_supply_units = 1


    # actually create and add the demand units
    for i in range(n_demand_units):
        id = "0" + str(i + 1) if i < 9 else str(i + 1)
        world.add_unit_operator(id=f"storage_demand_operator_{id}")
        world.add_unit(
            id=f"storage_demand_{id}",
            unit_type="demand",
            unit_operator_id=f"storage_demand_operator_{id}",
            unit_params={
                "baseline_storage": 0,  # unit has no storage
                "max_power": 1000,  # max 1.000 kW demand
                "min_power": 0,  # no minimum demand
                "bidding_strategies": {"BatteryMarket": "llm_buy_strategy"},
                "bidding_params": {"baseline_storage": 0},  # baseline to compare with
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
                solar_generation_forecast=forecasts["solar_generation_forecast"],
            ),
        )

    # actually create and add the supply units
    for i in range(n_supply_units):
        id = "0" + str(i + 1) if i < 9 else str(i + 1)
        world.add_unit_operator(f"storage_provider_operator_{id}")
        world.add_unit(
            id=f"storage_provider_{id}",
            unit_type="storage",
            unit_operator_id=f"storage_provider_operator_{id}",
            unit_params={
                "max_power_charge": 1,  # max 1 kW charge
                "max_power_discharge": 1,  # max 1 kW discharge
                "max_soc": 20,  # max 20 kWh of storage capacity (equal to baseline)
                "min_soc": 0,  # no mimimum fill level
                "efficiency_charge": 0.975,  # charge and discharge to combine to 95% efficiency
                "efficiency_discharge": 0.975,
                "bidding_strategies": {"BatteryMarket": "llm_sell_strategy"},
                "bidding_params": {"baseline_storage": 20},  # baseline to compare with, should be equal to max_soc
                "technology": "battery_storage",
            },
            forecaster=NaiveForecast(
                index=index,
                availability=1,  # always available
                energy_demand=forecasts["demand"],
                solar_generation=forecasts["solar_generation_forecast"],
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
    # db_uri = "postgresql://assume:assume@localhost:5432/assume"
    db_uri = "sqlite:///assume_db"
    world = World(database_uri=db_uri, log_level="ERROR")
    init(world)
    logging.getLogger("highs").setLevel(logging.WARNING)  # suppress gurobipy logs
    world.run()