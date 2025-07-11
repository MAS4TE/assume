# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import random
from datetime import timedelta
from itertools import groupby
from operator import itemgetter

import pyomo.environ as pyo

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)


def calculate_meta(accepted_supply_orders, accepted_demand_orders, product):
    supply_volume = sum(map(itemgetter("accepted_volume"), accepted_supply_orders))
    demand_volume = -sum(map(itemgetter("accepted_volume"), accepted_demand_orders))
    prices = list(map(itemgetter("accepted_price"), accepted_supply_orders)) or [0]
    # can also be self.marketconfig.maximum_bid..?
    duration_hours = (product[1] - product[0]) / timedelta(hours=1)
    avg_price = 0
    if supply_volume:
        weighted_price = [
            order["accepted_volume"] * order["accepted_price"]
            for order in accepted_supply_orders
        ]
        avg_price = sum(weighted_price) / supply_volume
    return {
        "supply_volume": supply_volume,
        "demand_volume": demand_volume,
        "demand_volume_energy": demand_volume * duration_hours,
        "supply_volume_energy": supply_volume * duration_hours,
        "price": avg_price,
        "max_price": max(prices),
        "min_price": min(prices),
        "node": None,
        "product_start": product[0],
        "product_end": product[1],
        "only_hours": product[2],
    }



class BatteryClearing(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def validate_orderbook(
        self, orderbook: Orderbook, agent_addr
    ) -> None:
        c_rates = self.marketconfig.param_dict["allowed_c_rates"]
        for order in orderbook:
            if order["c_rate"] not in c_rates:
                raise ValueError(f"{order['c_rate']} is not in {c_rates}")

        super().validate_orderbook(orderbook, agent_addr)

    def set_model_restrictions(self) -> None:
        """Sets the model restrictions."""

        def restrict_product_balance(model, storage_type):
            total_ask = sum(
                ask.product.components.get(storage_type, 0)
                * self.model.ask_volume[ask.uuid, ask.product.id]
                for ask in self.asks
            )
            total_bid = sum(
                bid.product.components.get(storage_type, 0)
                * self.model.bid_volume[bid.uuid, bid.product.id]
                for bid in self.bids
            )

            return total_ask >= total_bid

        self.model.restrict_product_balance = pyo.Constraint(
            self.storage_types, rule=restrict_product_balance
        )

    def set_model_objective(self) -> None:
        """Sets the model objective function."""

        supply_costs = sum(
            self.model.ask_volume[ask.uuid, ask.product.id]
            * (ask.price + 1e-3 * random())
            for ask in self.asks
        )
        demand_costs = sum(
            self.model.bid_volume[bid.uuid, bid.product.id] * bid.price
            for bid in self.bids
        )

        bez_menge = 1e-3 * (
            sum(self.model.bid_volume[bid.uuid, bid.product.id] for bid in self.bids)
            + sum(self.model.ask_volume[ask.uuid, ask.product.id] for ask in self.asks)
        )

        self.model.objective = pyo.Objective(
            expr=demand_costs - supply_costs + bez_menge, sense=pyo.maximize
        )

    def add_bids(self) -> None:
        """Creates bid volume variable for each bidder and product."""

        # create indexed var
        self.model.bid_volume = pyo.Var(
            [bid.uuid for bid in self.bids],
            self.bid_products,
            domain=pyo.NonNegativeReals,
        )

        # set upper bound
        for bid in self.bids:
            self.model.bid_volume[bid.uuid, bid.product.id].setub(bid.volume)

    def add_asks(self) -> None:
        """Creates ask volume variable for each asker and product."""

        # create indexed var
        self.model.ask_volume = pyo.Var(
            [ask.uuid for ask in self.asks],
            self.asked_products,
            domain=pyo.NonNegativeReals,
        )

        # set upper bound
        for ask in self.asks:
            self.model.ask_volume[ask.uuid, ask.product.id].setub(ask.volume)

    def gather_bidders(self) -> list[str]:
        """Gather all bidders in the market.

        Returns:
            list[Bid]: List of bidders
        """
        return list(set([bid.participant_id for bid in self.bids]))

    def gather_askers(self):
        """Gather all askers in the market.

        Returns:
            list[Ask]: List of askers
        """
        return list(set([ask.participant_id for ask in self.asks]))

    def gather_storage_types(self) -> list:
        """Gather all storage types in the market.

        Returns:
            list: List of storage types
        """

        all_storage_types = []
        for product in self.tradeable_products:
            product_storage_types = list(product.components.keys())
            all_storage_types.extend(product_storage_types)

        all_storage_types = list(set(all_storage_types))

        return all_storage_types

    def calculate_clearing_price(self) -> float:
        """Calculates the clearing price as the highest price of awarded asks or bids.

        Returns:
            float: The clearing price.
        """
        # Get all awarded ask prices where volume > 0
        awarded_ask_prices = [
            ask.price
            for ask in self.asks
            if self.model.ask_volume[ask.uuid, ask.product.id].value > 0
        ]

        # Get all awarded bid prices where volume > 0
        awarded_bid_prices = [
            bid.price
            for bid in self.bids
            if self.model.bid_volume[bid.uuid, bid.product.id].value > 0
        ]

        # Combine all awarded prices
        all_awarded_prices = awarded_ask_prices + awarded_bid_prices

        if not all_awarded_prices:
            raise ValueError("No trades occurred, clearing price cannot be determined.")

        # Return the highest awarded price
        return max(all_awarded_prices)

    def get_accepted_asks(self, clearing_price: float) -> list[AcceptedAsk]:
        """Retrieves accepted asks.

        Returns:
            list[AcceptedAsk]: The accepted asks.
        """

        accepted_asks = []
        for ask in self.asks:
            traded_volume = value(self.model.ask_volume[ask.uuid, ask.product.id])

            if traded_volume > 0:
                accepted_ask = AcceptedAsk.from_ask(
                    ask=ask,
                    accepted_volume=traded_volume,
                    accepted_price=clearing_price,
                )

                accepted_asks.append(accepted_ask)

        return accepted_asks

    def get_accepted_bids(self, clearing_price: float) -> list[AcceptedBid]:
        """Retrieves accepted bids.

        Returns:
            list[AcceptedBid]: The accepted bids.
        """

        accepted_bids = []
        for bid in self.bids:
            traded_volume = value(self.model.bid_volume[bid.uuid, bid.product.id])

            if traded_volume > 0:
                accepted_bid = AcceptedBid.from_bid(
                    bid=bid,
                    accepted_volume=traded_volume,
                    accepted_price=clearing_price,
                )

                accepted_bids.append(accepted_bid)

        return accepted_bids


    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        """
        Performs electricity market clearing using a pay-as-clear mechanism. This means that the clearing price is the
        highest price that is still accepted. The clearing price is the same for all accepted orders.

        Args:
            orderbook (Orderbook): the orders to be cleared as an orderbook
            market_products (list[MarketProduct]): the list of products which are cleared in this clearing

        Returns:
            tuple: accepted orderbook, rejected orderbook and clearing meta data
        """
        market_getter = itemgetter("start_time", "end_time", "c_rate")
        accepted_orders: Orderbook = []
        rejected_orders: Orderbook = []
        clear_price = 0
        meta = []
        orderbook.sort(key=market_getter)
        # for each start and end of market products, we have all combinations of allowed c_rates
        from itertools import product

        
        # create cartesian product, unwrap into list and append to it    
        products: list[dict] = [[*x, y] for x, y in product(self.marketconfig.param_dict["allowed_c_rates"], market_products)]

        supply_orders = [x for x in orderbook if x["volume"] > 0]
        demand_orders = [x for x in orderbook if x["volume"] < 0]

        import uuid
        product_ids = {uuid.uuid4(): product in products}
        for product, product_orders in groupby(orderbook, market_getter):
            accepted_demand_orders: Orderbook = []
            accepted_supply_orders: Orderbook = []
            product_orders = list(product_orders)
            if product["c_rate"] not in self.marketconfig.param_dict["allowed_c_rates"]:
                rejected_orders.extend(product_orders)
                # logger.debug(f'found unwanted bids for {product} should be {market_products}')
                continue

            # hier bin ich mir sicher, dass alle orders in product_orders die selbe c_rate haben

            supply_orders = [x for x in product_orders if x["volume"] > 0]
            demand_orders = [x for x in product_orders if x["volume"] < 0]
            # volume 0 is ignored/invalid

            # Sort supply orders by price with randomness for tie-breaking
            supply_orders.sort(key=lambda x: (x["price"], random.random()))

            # Sort demand orders by price in descending order with randomness for tie-breaking
            demand_orders.sort(
                key=lambda x: (x["price"], random.random()), reverse=True
            )

        self.asked_products = products
        self.bid_products = products
        self.askers = self.gather_askers()
        self.bidders = self.gather_bidders()
        
        self.storage_types = self.gather_storage_types()

        self.model = pyo.ConcreteModel()

        
        # create indexed var
        self.model.bid_volume = pyo.Var(
            [order["bid_id"] for order in demand_orders],
            products,
            domain=pyo.NonNegativeReals,
        )

        # set upper bound
        for order in demand_orders:
            self.model.bid_volume[order["bid_id"]].setub(order["volume"])

        self.add_asks()

        self.set_model_objective()

        self.set_model_restrictions()

        # TODO chriko97 run model here:
        solver = pyo.SolverFactory(solver)
        results = solver.solve(self.model, tee=False)
        # TODO get output from results and set this in each incoming bid

        # if demand is fulfilled, we do have some additional supply orders
        # these will be rejected
        for order in product_orders:
            # if the order was not accepted partially, it is rejected
            if not order.get("accepted_volume") and order not in rejected_orders:
                rejected_orders.append(order)

        # set clearing price - merit order - uniform pricing
        if accepted_supply_orders:
            clear_price = float(
                max(map(itemgetter("price"), accepted_supply_orders))
            )
        else:
            clear_price = 0

        accepted_product_orders = accepted_demand_orders + accepted_supply_orders
        for order in accepted_product_orders:
            order["accepted_price"] = clear_price
        accepted_orders.extend(accepted_product_orders)

        # set accepted volume to 0 and price to clear price for rejected orders
        for order in rejected_orders:
            order["accepted_volume"] = 0
            order["accepted_price"] = clear_price

        meta.append(
            calculate_meta(
                accepted_supply_orders,
                accepted_demand_orders,
                product,
            )
        )

        # write network flows here if applicable
        flows = []

        return accepted_orders, rejected_orders, meta, flows

