# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import timestamp2datetime

log = logging.getLogger(__name__)


def init(world, n=1):
    start_tz = datetime.now() + rd(minutes=1, second=0, microsecond=0)
    # start should be UTC
    start = timestamp2datetime(start_tz.timestamp())
    end = start + rd(hours=1)

    index = FastIndex(start, end, freq="m")
    simulation_id = "world_realtime"

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=1,
        simulation_id=simulation_id,
        # set real_time to true here
        real_time=True,
    )
    log.info("Starting %s from %s to %s", simulation_id, start, end)

    marketdesign = [
        MarketConfig(
            market_id="EOM",
            opening_hours=rr.rrule(
                rr.MINUTELY,
                interval=1,
                dtstart=start,
                until=end,
                cache=True,
                bysecond=0,
            ),
            opening_duration=timedelta(minutes=1),
            market_mechanism="pay_as_clear",
            market_products=[
                MarketProduct(timedelta(minutes=1), 1, timedelta(minutes=1))
            ],
            additional_fields=["block_id", "link", "exclusive_id"],
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
            "bidding_strategies": {"EOM": "naive_eom"},
            "technology": "demand",
        },
        NaiveForecast(index, demand=1000),
    )

    nuclear_forecast = NaiveForecast(index, availability=1, fuel_price=3, co2_price=0.1)
    for i in range(n):
        world.add_unit_operator(f"my_operator{i}")
        world.add_unit(
            f"nuclear{i}",
            "power_plant",
            f"my_operator{i}",
            {
                "min_power": 200 / n,
                "max_power": 1000 / n,
                "bidding_strategies": {"EOM": "naive_eom"},
                "technology": "nuclear",
            },
            nuclear_forecast,
        )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri, log_level="INFO")
    init(world)
    world.run()
