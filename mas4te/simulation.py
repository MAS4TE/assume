import logging
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

from mas4te_bidding_strategy import LLMStrategy
# from mas4te_clearing_mechanism import BatteryClearing

log = logging.getLogger(__name__)


def init(world: World, n=1):
    start = datetime(2025, 1, 1)
    end = datetime(2025, 3, 1)

    index = FastIndex(start, end, freq="h")
    simulation_id = "mas4te_simulation"

    world.bidding_strategies["llm_strategy"] = LLMStrategy
    # TODO implement BatteryClearing
    # world.clearing_mechanisms["battery_clearing"] = BatteryClearing

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=simulation_id,
    )

    marketdesign = [
        MarketConfig(
            market_id="BatteryMarket",
            opening_hours=rr.rrule(
                rr.WEEKLY, interval=1, dtstart=start, until=end, cache=True
            ),
            opening_duration=timedelta(hours=1),
            market_mechanism="pay_as_clear",
            product_type="battery",
            market_products=[MarketProduct(
                duration=timedelta(hours=24*7),
                count=1,
                first_delivery=timedelta(hours=12))],
        )
    ]

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    world.add_unit_operator(id="storage_demand_operator")
    world.add_unit(
        id="storage_demand_01",
        unit_type="demand",
        unit_operator_id="storage_demand_operator",
        unit_params={
            "min_power": 0,
            "max_power": 1000,
            "bidding_strategies": {"BatteryMarket": "llm_strategy"},
            "technology": "demand",
        },
        forecaster=NaiveForecast(index=index, demand=1),
    )

    world.add_unit_operator("storage_provider_operator")
    world.add_unit(
        id="storage_provider_01",
        unit_type="storage",
        unit_operator_id="storage_provider_operator",
        unit_params={
            "max_power_charge": 1,
            "max_power_discharge": 1,
            "max_soc": 10,
            "min_soc": 0,
            "efficiency_charge": 0.8,
            "efficiency_discharge": 0.85,
            "bidding_strategies": {"BatteryMarket": "llm_strategy"},
            "technology": "battery_storage",
        },
        forecaster=NaiveForecast(index=index, availability=1, demand=0, fuel_price=0, co2_price=0),
    )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    init(world)
    world.run()
