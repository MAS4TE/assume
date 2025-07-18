# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import requests
from pricing_framework import PricingFramework, Storage

from assume.common.base import BaseStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product


class LLMStrategy(BaseStrategy):
    """
    A strategy that uses a Large Language Model (LLM) for a storage buyer.

    Params:
        llm_api_url (str): The URL of the LLM API to use for generating bids.
    """

    def __init__(self, llm_api_url=None, baseline_storage=0, *args, **kwargs):
        super().__init__()
        self.baseline_storage = baseline_storage
        self.api_url = llm_api_url
        self.headers = {"Content-Type": "application/json"}

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
    ) -> dict[float, float]:
        """Calculates price recommendations for specific storage volumes depending on forecasted price, energy demand and solar generation for a specific unit.

        Args:
            unit (SupportsMinMaxCharge): The unit to calculate bids for.
            product (Product): The product for which to calculate the price recommendations.
            storages_to_calculate (list[Storage]): List of storage volumes to calculate the worth for.

        Returns:
        dict: monetary worth (marginal costs) of the storage volume to the unit operator with volume as key and worth as value.
        """

        # start of by reading in / generating the data for demand and costs
        demand_timeseries = unit.forecaster["demand"]  # build_demand_timeseries(unit)
        solar_generation = unit.forecaster["solar_generation"]

        # prices are forecasted in series but need to be in DataFrame format for the optimizer
        wholesale_prices = unit.forecaster[
            "wholesale_price"
        ]  # build_wholesale_prices(unit)
        eeg_prices = unit.forecaster["eeg_price"]  # build_eeg
        community_prices = unit.forecaster[
            "community_price"
        ]  # build_community_prices(unit)
        grid_prices = unit.forecaster["grid_price"]  # build_grid_prices(unit)
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
                id=0, c_rate=1, volume=self.baseline_storage, efficiency=0.95
            ),
            prices=prices,
            solar_generation=solar_generation,
            demand=demand_timeseries,
            storage_use_cases=["eeg", "wholesale", "community", "home"],
        )
        pricer.optimize(solver="gurobi")
        baseline_cost = pricer.model.objective()

        storages_values[self.baseline_storage] = baseline_cost

        for storage in storages_to_calculate:

            # create the optimizer
            pricer = PricingFramework(
                storage=storage,
                prices=prices,
                solar_generation=solar_generation,
                demand=demand_timeseries,
                storage_use_cases=["eeg", "wholesale", "community", "home"],
            )

            # run the optimization
            pricer.optimize(solver="gurobi")

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


class LLMBuyStrategy(LLMStrategy):
    """A strategy that uses a Large Language Model (LLM) for a storage buyer."""

    def __init__(self, llm_api_url=None, baseline_storage=0, *args, **kwargs):
        super().__init__(llm_api_url, baseline_storage, *args, **kwargs)

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
                unit=unit, product=product, storages_to_calculate=None
            )

        print("#####################################################")
        print("Volumes and their values for buyer:")
        for volume, value in volumes_values.items():
            print(f"Volume to buy: {volume}, Marginal worth (max. to pay): {value}")
        print("#####################################################")
        print(
            "This input is only here to pause the simulation and allow you to see the recommendations"
        )
        print("Input something to continue...")
        input()

        #####################################################
        #               LLM CALL GOES HERE                  #
        #   TO CHOOSE FROM RECOMMENDATIONS OR ALTER THEM    #
        #####################################################
        llm_price_recommendation = 10
        llm_volume_recommendation = -1

        bids = []
        for product in product_tuples:
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": llm_price_recommendation,
                    "volume": llm_volume_recommendation,
                }
            )

        return bids


class LLMSellStrategy(LLMStrategy):
    """A strategy that uses a Large Language Model (LLM) for a storage seller."""

    def __init__(self, llm_api_url=None, baseline_storage=0, *args, **kwargs):
        super().__init__(llm_api_url, baseline_storage, *args, **kwargs)

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
                unit=unit, product=product, storages_to_calculate=None
            )

        print("#####################################################")
        print("Volumes and their values for seller:")
        for volume, value in volumes_values.items():
            if volume > self.baseline_storage:
                continue
            print(f"Volume to sell: {self.baseline_storage - volume}, Marginal worth (min. to receive): {-value}")
        print("#####################################################")
        print(
            "This input is only here to pause the simulation and allow you to see the recommendations"
        )
        print("Input something to continue...")
        input()

        #####################################################
        #               LLM CALL GOES HERE                  #
        #   TO CHOOSE FROM RECOMMENDATIONS OR ALTER THEM    #
        #####################################################
        llm_price_recommendation = 1
        llm_volume_recommendation = 1

        bids = []
        for product in product_tuples:
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": llm_price_recommendation,
                    "volume": llm_volume_recommendation,
                }
            )

        return bids

