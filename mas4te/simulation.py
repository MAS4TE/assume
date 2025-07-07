import logging
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

from mas4te_bidding_strategy import LLMBuyStrategy
# from mas4te_clearing_mechanism import BatteryClearing

import pandas as pd

log = logging.getLogger(__name__)


def init(world: World, n=1):

    # set start and end date
    start = datetime(2025, 1, 1)
    end = datetime(2025, 3, 1)

    # create index
    index = FastIndex(start, end, freq="15min")

    # set simulation ID
    simulation_id = "mas4te_simulation"

    # add possible bidding strategies
    world.bidding_strategies["llm_buy_strategy"] = LLMBuyStrategy

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
                rr.WEEKLY, interval=1, dtstart=start, until=end, cache=True # weekly battery market with the next week tradeable
            ),
            opening_duration=timedelta(hours=1),
            market_mechanism="pay_as_clear",
            product_type="battery",
            market_products=[MarketProduct(
                duration=timedelta(hours=24*7),         # each product (storage rent) will be 1 week
                count=1,                                # we will only trade the next week, not any week after that
                first_delivery=timedelta(hours=12))],   # delivery will take place 12 hours after market close
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
    # TODO: replace with actual data reading
    demand = pd.read_csv()
    prices = pd.read_csv()
    solar_generation = pd.read_csv()

    # actually create and add the demand unit
    world.add_unit_operator(id="storage_demand_operator")
    world.add_unit(
        id="storage_demand_01",
        unit_type="demand",
        unit_operator_id="storage_demand_operator",
        unit_params={
            "baseline_storage": 0,  # unit has no storage
            "max_power": 1000,      # max 1.000 kW demand
            "min_power": 0,         # no minimum demand
            "bidding_strategies": {"BatteryMarket": "llm_buy_strategy"},
            "technology": "demand",
        },
        forecaster=NaiveForecast(
            index=index,
            demand=1),
    )

    # create and add provider (sell) unit
    world.add_unit_operator("storage_provider_operator")
    world.add_unit(
        id="storage_provider_01",
        unit_type="storage",
        unit_operator_id="storage_provider_operator",
        unit_params={
            "max_power_charge": 1,          # max 1 kW charge
            "max_power_discharge": 1,       # max 1 kW discharge
            "max_soc": 10,                  # max 10 kWh of storage capacity
            "min_soc": 0,                   # no mimimum fill level
            "efficiency_charge": 0.975,     # charge and discharge to combine to 95% efficiency
            "efficiency_discharge": 0.975,
            "bidding_strategies": {"BatteryMarket": "llm_sell_strategy"},
            "technology": "battery_storage",
        },
        forecaster=NaiveForecast(
            index=index,
            availability=1, # always available
            demand=0,       # no battery demand
            fuel_price=0,   # no fuel price
            co2_price=0),   # no co2 price
    )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    init(world)
    world.run()
