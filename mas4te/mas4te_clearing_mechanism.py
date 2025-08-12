# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import random
from datetime import timedelta
from itertools import groupby
from operator import itemgetter

import pyomo.environ as pyo

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook, Order
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
        allowed_c_rates = self.marketconfig.param_dict["allowed_c_rates"]
        for order in orderbook:
            if order["c_rate"] not in allowed_c_rates:
                raise ValueError(f"{order['c_rate']} is not in {allowed_c_rates}")

        super().validate_orderbook(orderbook, agent_addr)

    def set_model_restrictions(self, model: pyo.ConcreteModel) -> None:
        """Sets the model restrictions."""

        # Restrict the supply volume to be less than or equal to the demand volume
        model.restrict_product_balance = pyo.Constraint(
            sum(model.supply_volume[i] for i in model.supply_volume) <= \
            sum(model.demand_volume[i] for i in model.demand_volume)
        )

        model.restrict_prices = pyo.Constraint(
            model.supply_price[i] for i in model.supply_prices
        )

    def set_model_objective(self, model: pyo.ConcreteModel) -> None:
        """Sets the model objective function."""

        supply_costs = sum(model.supply_price[bid_id] * model.supply_volume[bid_id] for bid_id in model.supply_prices)
        demand_costs = sum(model.demand_price[bid_id] * model.demand_volume[bid_id] for bid_id in model.demand_prices)

        model.objective = pyo.Objective(
            expr=demand_costs - supply_costs, sense=pyo.maximize
        )

    def add_supply_vars(self, model: pyo.ConcreteModel, supply_orders: list[Order]) -> None:
        """Creates supply price & volume variable."""

        model.supply_volume = pyo.Var(
            [supply_order["bid_id"] for supply_order in supply_orders],
            domain=pyo.NonNegativeReals,
        )
        for supply_order in supply_orders:
            max_volume = abs(supply_order["volume"])
            model.supply_volume[supply_order["bid_id"]].setub(max_volume)

        model.supply_price = pyo.Var(
            [supply_order["bid_id"] for supply_order in supply_orders],
            domain=pyo.NonNegativeReals,
        )
        for supply_order in supply_orders:
            max_price = supply_order["price"]
            model.supply_price[supply_order["bid_id"]].setub(max_price)

    def add_demand_vars(self, model: pyo.ConcreteModel, demand_orders: list[Order]) -> None:
        """Creates demand price & volume variable."""

        model.demand_volume = pyo.Var(
            [demand_order["bid_id"] for demand_order in demand_orders],
            domain=pyo.NonNegativeReals,
        )
        for demand_order in demand_orders:
            max_volume = abs(demand_order["volume"])
            self.model.demand_volume[demand_order["bid_id"]].setub(max_volume)

        model.demand_price = pyo.Var(
            [demand_order["bid_id"] for demand_order in demand_orders],
            domain=pyo.NonNegativeReals,
        )
        for demand_order in demand_orders:
            max_price = demand_order["price"]
            model.demand_price[demand_order["bid_id"]].setub(max_price)

    def solve_model(self, model: pyo.ConcreteModel, solver: str = "highs") -> None:
        """Solves the model using the specified solver."""

        solver = pyo.SolverFactory(solver)
        results = solver.solve(model, tee=False)

        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise ValueError("Model could not be solved optimally.")

        # Log the results
        logger.info("Model solved successfully.")
        logger.debug(f"Objective value: {pyo.value(model.objective)}")

    def get_accepted_rejected_orders(
        self, model: pyo.ConcreteModel, supply_orders: list[Order], demand_orders: list[Order]
    ) -> tuple[list[Order], list[Order]]:

        accepted_orders = []
        rejected_orders = []
        for supply_order in supply_orders:
            volume = model.supply_volume[supply_order["bid_id"]].value
            supply_order["accepted_volume"] = volume

            if volume > 0:
                accepted_orders.append(supply_order)
            else:
                rejected_orders.append(supply_order)

        for demand_order in demand_orders:
            volume = model.demand_volume[demand_order["bid_id"]].value
            demand_order["accepted_volume"] = volume

            if volume > 0:
                accepted_orders.append(demand_order)
            else:
                rejected_orders.append(demand_order)

        return accepted_orders, rejected_orders

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

    def clear(
            self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:

        # get demand and supply orders from orderbook
        demand_orders = [x for x in orderbook if x["volume"] < 0]
        supply_orders = [x for x in orderbook if x["volume"] > 0]

        # create pyomo model for solving
        model = pyo.ConcreteModel()

        # add supply and demand variables to the model
        self.add_supply_vars(model, supply_orders)
        self.add_demand_vars(model, demand_orders)

        # add restrictions to the model
        self.set_model_restrictions(model)

        # set the model objective
        self.set_model_objective(model)

        # run optimization (clearing)
        self.solve_model(model, solver="highs")

        # get accepted orders
        accepted_orders, rejected_orders = self.get_accepted_rejected_orders(
            model, supply_orders, demand_orders
        )

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

