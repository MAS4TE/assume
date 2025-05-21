import logging
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

from mas4te_bidding_strategy import LLMStrategy
#from mas4te_clearing_mechanism import BatteryClearing

log = logging.getLogger(__name__)


def init(world: World, n=1):
    start = datetime(2025, 1, 1)
    end = datetime(2025, 3, 1)

    index = FastIndex(start, end, freq="h")
    simulation_id = "mas4te_simulation"

    world.bidding_strategies["llm_strategy"] = LLMStrategy
    #world.clearing_mechanisms["battery_clearing"] = BatteryClearing

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=simulation_id,
    )

    marketdesign = [
        # flexibility market which clears
        MarketConfig(
            market_id="BatteryMarket",
            opening_hours=rr.rrule(
                rr.HOURLY, interval=24, dtstart=start, until=end, cache=True
            ),
            opening_duration=timedelta(hours=1),
            market_mechanism="pay_as_clear",
            product_type="battery",
            market_products=[MarketProduct(timedelta(hours=1), 24, timedelta(hours=1))],
            additional_fields=["c_rate", "and_link", "xor_link"],
            param_dict={"allowed_c_rates": [0.5, 1, 1.5]}
        )
    ]

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    world.add_unit_operator("my_demand")
    world.add_unit(
        "demand1",
        "demand",
        "my_demand",
        # the unit_params have no hints
        {
            "min_power": 0,
            "max_power": 1000,
            "bidding_strategies": {"BatteryMarket": "llm_strategy", "EOM": "llm_strategy"},
            "technology": "demand",
        },
        NaiveForecast(index, demand=1000),
    )

    gas_forecast = NaiveForecast(index, availability=1, fuel_price=3, co2_price=0.1)
    for i in range(n):
        world.add_unit_operator(f"my_operator{i}")
        world.add_unit(
            f"gas{i}",
            "power_plant",
            f"my_operator{i}",
            {
                "min_power": 200 / n,
                "max_power": 1000 / n,
                "bidding_strategies": {"BatteryMarket": "naive_eom", "EOM": "naive_eom"},
                "technology": "natural gas",
            },
            gas_forecast,
        )
    
    world.add_unit_operator("storage_operator")
    world.add_unit(
        "storage1",
        "storage",
        "storage_operator",
        {
            "max_power_charge": 1e3,
            "max_power_discharge": 1e3,
            "max_soc": 1e3, # kWh
            "min_soc": 0,
            "efficiency_charge": 0.8,
            "efficiency_discharge": 0.85,
            "bidding_strategies": {"BatteryMarket": "flexable_eom_storage", "EOM": "flexable_eom_storage"},
            "technology": "battery_storage",
        },
        NaiveForecast(index, availability=1, fuel_price=0.2, co2_price=0),
    )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    init(world)
    world.run()
