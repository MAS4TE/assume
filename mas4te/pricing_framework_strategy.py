# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import random
from datetime import datetime

import pandas as pd
import requests
from pricing_framework import PricingFramework, Storage

from assume.common.base import BaseStrategy, BaseUnit, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import get_supported_solver


class PricingFrameworkStrategy(BaseStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def build_storages_to_calculate(self):
        """Builds a list of storage volumes to calculate worth for.

        Returns:
            list[Storage]: List of Storage objects with different volumes.
        """
        # Example: Create storages with volumes from 0 to 1000 in steps of 100
        storages = [
            Storage(id=i, volume=i, c_rate=1, efficiency=0.95) for i in range(1, 15)
        ]

        storages += [
            Storage(id=i, volume=i * 5, c_rate=1, efficiency=0.95) for i in range(3, 11)
        ]

        return storages

    def calculate_storage_values(
        self,
        unit: SupportsMinMaxCharge,
        product: Product,
        storages_to_calculate: list[Storage],
        baseline_storage,
    ) -> dict[float, float]:
        """Calculates price recommendations for specific storage volumes depending on forecasted price, energy demand and solar generation for a specific unit.

        Args:
            unit (SupportsMinMaxCharge): The unit to calculate bids for.
            product (Product): The product for which to calculate the price recommendations.
            storages_to_calculate (list[Storage]): List of storage volumes to calculate the worth for.

        Returns:
        dict: monetary worth (marginal costs) of the storage volume to the unit operator with volume as key and worth as value.
        """

        start_time = datetime(2023, 1, 1, hour=13)
        end_time = datetime(2023, 1, 8, hour=13)

        # start of by reading in / generating the data for demand and costs
        demand_timeseries = unit.forecaster["energy_demand"]#.loc[start_time:end_time, :]
        solar_gen = unit.forecaster["solar_gen"]#.loc[start_time:end_time, :]

        # prices are forecasted in series but need to be in DataFrame format for the optimizer
        wholesale_prices = unit.forecaster[
            "wholesale_price"
        ]#.loc[start_time:end_time]  # build_wholesale_prices(unit)
        eeg_prices = unit.forecaster["eeg_price"]#.loc[start_time:end_time, :]  # build_eeg
        community_prices = unit.forecaster[
            "community_price"
        ]#.loc[start_time:end_time]  # build_community_prices(unit)
        grid_prices = unit.forecaster["grid_price"]#.loc[start_time:end_time, :]  # build_grid_prices(unit)

        prices = pd.DataFrame(
            data={
                "wholesale": wholesale_prices,
                "eeg": eeg_prices,
                "community": community_prices,
                "grid": grid_prices,
            },
        )

        # build storages to optimize for
        # change storages that should be calculated here!
        if not storages_to_calculate:
            storages_to_calculate = self.build_storages_to_calculate()

        # dictionary to hold the worth of each storage in
        # with volume as key and worth as value
        storages_values = {}

        # get baseline optimization, to know how much prosumer has to pay with current setup
        pricer = PricingFramework(
            storage=Storage(
                id=0, c_rate=1, volume=baseline_storage, efficiency=0.95
            ),
            prices=prices,
            solar_generation=solar_gen,
            demand=demand_timeseries,
            storage_use_cases=["eeg", "wholesale", "community", "home"],
        )
        pricer.optimize(solver=get_supported_solver("gurobi"))
        baseline_cost = pricer.model.objective()

        storages_values[baseline_storage] = baseline_cost

        for storage in storages_to_calculate:

            # create the optimizer
            pricer = PricingFramework(
                storage=storage,
                prices=prices,
                solar_generation=solar_gen,
                demand=demand_timeseries,
                storage_use_cases=["eeg", "wholesale", "community", "home"],
            )

            # run the optimization
            pricer.optimize(solver=get_supported_solver("gurobi"))

            # minimum cost in this scenario is the objective of the optimization model
            # we're optimizing energy dispatch to potential storage to minimize costs, thus
            # we're not calculating optimal storage size or the worth of the storage, but the costs
            # associated with the storage volume
            # to get the worth of the storage, we need to compare the costs with the baseline costs, e. g.
            # the costs of the current setup without storage (buyer side) or with a specific storage volume (seller side)
            minimum_cost = pricer.model.objective()

            # worth of the storage is the difference between the baseline cost and the minimum cost
            value = minimum_cost - baseline_cost

            # add to storage_value dictionary
            storages_values[storage.volume] = value

        return storages_values

    def run_prompt(
        self, prompt: str, model="Mistral-7B-Instruct-v0.3-Q4_K_M", max_tokens=1000
    ):
        data = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
        response = requests.post(self.api_url, headers=self.headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("text", "")

class BuyStrategy(PricingFrameworkStrategy):
    """A strategy that uses a Large Language Model (LLM) for a storage buyer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """Calculates the value of multiple storage volumes for a predicted demand and price timeseries via linear optimization.

        Args:
            unit (SupportsMinMaxCharge): The unit to calculate bids for.
            market_config (MarketConfig): The market configuration to use.
            product_tuples (list[Product]): The list of products to calculate bids for.

        Returns:
            Orderbook: The calculated order book with bids.
        """

        # print(dir(unit))

        # iterate over each product (which is only one in phase 1)
        for product in product_tuples:
            # get price recommendations for the product
            # for given storages to calculate the wort (value) for
            # if the LLM should provide some storages that should be calculated, it can be passed here
            # otherwise change the default storages in the build_storages_to_calculate method
            # this function is just a wrapper for the pricing framework
            volumes_values = self.calculate_storage_values(
                unit=unit, product=product, storages_to_calculate=None, baseline_storage=0,
            )

        choosen_volume = random.choice(list(volumes_values.keys()))
        resulting_price = float(volumes_values[choosen_volume] / choosen_volume)

        bids = []
        for product in product_tuples:
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": resulting_price,
                    "volume": -choosen_volume,
                    "c_rate": 1
                }
            )

        return bids


    def calculate_reward(
        self,
        unit: BaseUnit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward for the given unit.

        Args:
            unit (BaseUnit): The unit.
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.
        """
        # here we can learn something from our previous biddings
        # TODO Bea
        self.prompts = ...
        self.accepted_orders = orderbook


class SellStrategy(PricingFrameworkStrategy):
    """A strategy that uses a Large Language Model (LLM) for a storage seller."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """Calculates the value of multiple storage volumes for a predicted demand and price timeseries via linear optimization.

        Args:
            unit (SupportsMinMaxCharge): The unit to calculate bids for.
            market_config (MarketConfig): The market configuration to use.
            product_tuples (list[Product]): The list of products to calculate bids for.

        Returns:
            Orderbook: The calculated order book with bids.
        """

        # iterate over each product (which is only one in phase 1)
        for product in product_tuples:
            # get price recommendations for the product
            # for given storages to calculate the wort (value) for
            # if the LLM should provide some storages that should be calculated, it can be passed here
            # otherwise change the default storages in the build_storages_to_calculate method
            # this function is just a wrapper for the pricing framework
            volumes_values = self.calculate_storage_values(
                unit=unit, product=product, storages_to_calculate=None, baseline_storage=random.randrange(2, 50)
            )

        choosen_volume = random.choice(list(volumes_values.keys()))
        resulting_price = float(volumes_values[choosen_volume] / choosen_volume)

        bids = []
        for product in product_tuples:
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": resulting_price,
                    "volume": choosen_volume,
                    "c_rate": 1
                }
            )
        return bids

