from assume.common.base import BaseStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
import requests
import pandas as pd
from pricing_framework import Storage, Optimizer


def get_price_recommendations(
          unit: SupportsMinMaxCharge,
          product: Product,
) -> dict[float: float]:
    """Calculates price recommendations for specific storage volumes depending on forecasted price, energy demand and solar generation for a specific unit.

    Args:
        unit (SupportsMinMaxCharge): The unit to calculate bids for.
        product (Product): The product for which to calculate the price recommendations.

    Returns:
       dict: monetary worth (marginal costs) of the storage volume to the unit operator with volume as key and worth as value.
    """

    # get start and end time from product
    start, end = product[0], product[1]

    # start of by reading in / generating the data for demand and costs
    demand_timeseries = unit.forecaster["demand"][start:end] # build_demand_timeseries(unit)
    pricing_timeseries = unit.forecaster["price"]
    solar_generation = unit.forecaster["solar_generation"]

    # build storages to optimize for
    # TODO: Implement func or pass via parameter
    storages_to_calculate = build_storages_to_calculate()

    # baseline storage volume
    baseline_storage = unit.baseline_storage

    # get baseline optimization, to know how much prosumer has to pay with current setup
    optimizer = Optimizer(
        storage=baseline_storage,
        prices=pricing_timeseries,
        solar_generation=solar_generation,
        demand=demand_timeseries,
        storage_use_cases=["eeg", "wholesale", "community", "home"],
    )
    optimizer.optimize(solver="gurobi")
    baseline_cost = optimizer.model.objective()

    # dictionary to hold the worth of each storage in
    # with volume as key and worth as value
    storages_worth = {}

    for storage in storages_to_calculate:
        # create the optimizer
        optimizer = Optimizer(
            storage=storage,
            prices=pricing_timeseries,
            solar_generation=solar_generation,
            demand=demand_timeseries,
            storage_use_cases=["eeg", "wholesale", "community", "home"],
        )

        # run the optimization
        optimizer.optimize(solver="gurobi")

        # minimum cost in this scenario is the objective of the optimization model
        # we're optimizing energy dispatch to potential storage to minimize costs, thus
        # we're not calculating optimal storage size or the worth of the storage, but the costs
        # associated with the storage volume
        # to get the worth of the storage, we need to compare the costs with the baseline costs, e. g.
        # the costs of the current setup without storage (buyer side) or with a specific storage volume (seller side)
        minimum_cost = optimizer.model.objective()

        # worth of the storage is the difference between the baseline cost and the minimum cost
        worth = minimum_cost - baseline_cost

        # add to storage_worth dictionary
        storages_worth[storage.volume] = worth

    return storages_worth


class LLMBuyStrategy(BaseStrategy):
    """
    A strategy that uses a Large Language Model (LLM) for a storage buyer.

    Params:
        llm_api_url (str): The URL of the LLM API to use for generating bids.
    """
    def __init__(self, llm_api_url = None, *args, **kwargs ):
        super().__init__()
        self.api_url = llm_api_url
        self.headers = {"Content-Type": "application/json"}

    def run_prompt(self, prompt: str, model="Mistral-7B-Instruct-v0.3-Q4_K_M", max_tokens=1000):
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens
        }
        response = requests.post(self.api_url, headers=self.headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("text", "")

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
            volume_price_recommandations = get_price_recommendations(unit=unit, product=product)

        #####################################################
        #               LLM CALL GOES HERE                  #
        #####################################################

