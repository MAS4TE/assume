# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products, get_supported_solver
from mas4te.mas4te_clearing_mechanism import BatteryClearing
from mas4te.pricing_framework import Storage, PricingFramework

from .utils import create_orderbook, extend_orderbook

import pandas as pd

simple_dayahead_auction_config = MarketConfig(
    market_id="BatteryMarket",
    opening_hours=rr.rrule(
        rr.WEEKLY,
        interval=1,
        dtstart=datetime(2005, 6, 1),
        until=datetime(2005, 6, 14),
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


def test_market():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    # one wants to rent a battery of 1000kWh for this time interval - for a very high price
    orderbook = extend_orderbook(products, volume=-1000, price=3000)
    # someon has this volume for a price of 100
    orderbook = extend_orderbook(products, volume=1000, price=100, orderbook=orderbook)
    # someone has a volume of 900 for a price of 50
    orderbook = extend_orderbook(products, volume=900, price=50, orderbook=orderbook)

    for order in orderbook:
        order["c_rate"] = 1

    mr = BatteryClearing(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    import pandas as pd

    print(pd.DataFrame(mr.all_orders))
    print(pd.DataFrame(accepted))
    print(meta)

def test_edge_cases():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    orderbook = []
    for i in range(10):
        orderbook = extend_orderbook(products, volume=-100, price=3000, orderbook=orderbook)
    for i in range(10):
        orderbook = extend_orderbook(products, volume=100, price=90, orderbook=orderbook)
    orderbook = extend_orderbook(products, volume=900, price=50, orderbook=orderbook)

    for order in orderbook:
        order["c_rate"] = 1

    mr = BatteryClearing(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    import pandas as pd

    print(pd.DataFrame(mr.all_orders))
    print(pd.DataFrame(accepted))
    print(meta)

def test_pricing_framework():

    # buying 1 kWh for 1 €/kWh should equal to 3€ total
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=0, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [1, 1, 1]}),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([1, 1, 1])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    assert pricer.model.objective() == -3

    # buying 2 kWh for 0€/kWh and storing 1 kWh of this should equal 1€ total
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=1, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [0, 1, 1]}),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([1, 1, 1])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    assert pricer.model.objective() == -1

    # now we need 2 kWh at each timestep
    # on timestep=0, we can buy for 0€/kWh and should buy 3kWh
    # as we use 2 kWh during timestep=0 and use 1 kWh for timestep=1
    # total cost should be 2*0 + 1*1 + 2*1 = 3
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=1, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [0, 1, 1]}),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([2, 2, 2])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    assert pricer.model.objective() == -3

    # here we should gain 1€ from selling pv
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=0, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [1, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [1, 1, 1]}),
        solar_generation=pd.Series([1, 0, 0]),
        demand=pd.Series([0, 0, 0])
    )
    pricer.optimize(solver="gurobi")
    assert pricer.model.objective() == 1

    # same as above, but we can store PV and sell at
    # timestep=1 instead of timestep=0, as we can get 2€/kWh
    # in timestep=1
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=1, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [1, 2, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [1, 1, 1]}),
        solar_generation=pd.Series([1, 0, 0]),
        demand=pd.Series([0, 0, 0])
    )
    pricer.optimize(solver="gurobi")
    assert pricer.model.objective() == 2

    # charge from solar_generation in ts=0,1 and discharge at ts=2
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=2, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [5, 10, 20]}),
        solar_generation=pd.Series([1, 1, 0]),
        demand=pd.Series([0, 0, 2])
    )
    pricer.optimize(solver="gurobi")
    print(pricer.model.objective())
    assert pricer.model.objective() == 0
