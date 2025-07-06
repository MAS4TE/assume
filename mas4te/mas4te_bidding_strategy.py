from assume.common.base import BaseStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
import requests
import pandas as pd
from pricing_framework import Storage, Optimizer

# TODO: implement
def build_demand_timeseries(unit: SupportsMinMaxCharge) -> pd.DataFrame:
    """Builds a demand timeseries for the given unit.

    Returns:
        pd.DataFrame: The demand timeseries for the unit.
    """
    pass

# TODO: implement
def build_price_timeseries(unit: SupportsMinMaxCharge) -> pd.DataFrame:
    """Builds a price timeseries for the given unit.

    Args:
        unit (SupportsMinMaxCharge): The unit to build the price timeseries for.

    Returns:
        pd.DataFrame: The price timeseries for the unit.
    """
    pass

# TODO: implement
def build_solar_generation_timeseries(unit: SupportsMinMaxCharge) -> pd.DataFrame:
    """Builds a solar generation timeseries for the given unit.

    Args:
        unit (SupportsMinMaxCharge): The unit to build the solar generation timeseries for.

    Returns:
        pd.DataFrame: The solar generation timeseries for the unit.
    """
    pass

# TODO: implement
def build_storages_to_calculate() -> list[Storage]:
    """Builds a list of storages to calculate for.

    Returns:
        list[Storage]: The list of storages to calculate for.
    """
    # This is a placeholder implementation. Replace with actual logic to build storages.
    return [Storage(id=str(i), volume=i, c_rate=1, efficiency=0.9) for i in range(1, 10)]

# TODO: implement
def get_baseline_storage(unit: SupportsMinMaxCharge) -> Storage:
    """Gets the baseline storage for the given unit.

    Args:
        unit (SupportsMinMaxCharge): The unit to get the baseline storage for.

    Returns:
        Storage: The baseline storage for the unit.
    """
    # This is a placeholder implementation. Replace with actual logic to get baseline storage.
    return Storage(id="baseline", volume=0, c_rate=1, efficiency=0.9)

class LLMBuyStrategy(BaseStrategy):
    """
    A strategy that uses a Large Language Model (LLM) for a storage buyer.

    Params:
        llm_api_url (str): The URL of the LLM API to use for generating bids.
    """
    def __init__(self, llm_api_url, *args, **kwargs ):
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
        
        # start of by reading in / generating the data for demand and costs
        demand_timeseries = build_demand_timeseries(unit)
        pricing_timeseries = build_price_timeseries(unit)
        solar_generation = build_solar_generation_timeseries(unit)

        # build storages to optimize for
        storages_to_calculate = build_storages_to_calculate()

        # baseline storage volume
        baseline_storage = get_baseline_storage(unit)

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
            minimum_cost = optimizer.model.objective()

            # worth of the storage is the difference between the baseline cost and the minimum cost
            worth = minimum_cost - baseline_cost

            # add to storage_worth dictionary
            storages_worth[storage.volume] = worth
