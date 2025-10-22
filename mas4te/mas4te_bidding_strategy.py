# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import battery_utility_calculator as buc
import requests
from battery_utility_calculator import Storage

from assume.common.base import BaseStrategy, BaseUnit, SupportsMinMaxCharge
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

    def build_storages_to_calculate(self, baseline_storage: Storage):
        """Builds a list of storage volumes to calculate worth for.

        Returns:
            list[Storage]: List of Storage objects with different volumes.
        """
        storages = [
            Storage(id=i, volume=i, c_rate=1, efficiency=0.95) for i in range(1, 15)
        ]

        storages += [
            Storage(id=i, volume=i * 5, c_rate=1, efficiency=0.95) for i in range(3, 11)
        ]

        if baseline_storage.volume > 0:
            return [stor for stor in storages if stor.volume <= baseline_storage.volume]
        else:
            return storages

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

        bids = []
        storages_to_calculate = self.build_storages_to_calculate(
            baseline_storage=unit.baseline_storage
        )

        # iterate over each product (which is only one in phase 1)
        for product in product_tuples:
            start = product[0]
            end = product[1]

            storages_worth = buc.calculate_multiple_storage_worth(
                baseline_storage=unit.baseline_storage,
                storages_to_calculate=storages_to_calculate,
                demand=unit.forecaster["energy_demand"].as_pd_series(
                    start=start, end=end
                ),
                solar_generation=unit.forecaster["solar_gen"].as_pd_series(
                    start=start, end=end
                ),
                grid_prices=unit.forecaster["grid_price"].as_pd_series(
                    start=start, end=end
                ),
                eeg_prices=unit.forecaster["eeg_price"].as_pd_series(
                    start=start, end=end
                ),
                community_market_prices=unit.forecaster["community_price"].as_pd_series(
                    start=start, end=end
                ),
                wholesale_market_prices=unit.forecaster["wholesale_price"].as_pd_series(
                    start=start, end=end
                ),
                solver="gurobi",
            )
            bidding_curve = buc.calculate_bidding_curve(
                volumes_worth=storages_worth,
                buy_or_sell_side="buyer",
            )

            for idx, row in bidding_curve.iterrows():
                bids.append(
                    {
                        "start_time": product[0],
                        "end_time": product[1],
                        "only_hours": product[2],
                        "price": row["marginal_price"],
                        "volume": -row["volume"],
                        "c_rate": 1,
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

        bids = []
        storages_to_calculate = self.build_storages_to_calculate(
            baseline_storage=unit.baseline_storage
        )

        # iterate over each product (which is only one in phase 1)
        for product in product_tuples:
            start = product[0]
            end = product[1]

            storages_worth = buc.calculate_multiple_storage_worth(
                baseline_storage=unit.baseline_storage,
                storages_to_calculate=storages_to_calculate,
                demand=unit.forecaster["energy_demand"].as_pd_series(
                    start=start, end=end
                ),
                solar_generation=unit.forecaster["solar_gen"].as_pd_series(
                    start=start, end=end
                ),
                grid_prices=unit.forecaster["grid_price"].as_pd_series(
                    start=start, end=end
                ),
                eeg_prices=unit.forecaster["eeg_price"].as_pd_series(
                    start=start, end=end
                ),
                community_market_prices=unit.forecaster["community_price"].as_pd_series(
                    start=start, end=end
                ),
                wholesale_market_prices=unit.forecaster["wholesale_price"].as_pd_series(
                    start=start, end=end
                ),
                solver="gurobi",
            )
            bidding_curve = buc.calculate_bidding_curve(
                volumes_worth=storages_worth,
                buy_or_sell_side="seller",
            )

            for idx, row in bidding_curve.iterrows():
                bids.append(
                    {
                        "start_time": product[0],
                        "end_time": product[1],
                        "only_hours": product[2],
                        "price": row["marginal_price"],
                        "volume": row["volume"],
                        "c_rate": 1,
                    }
                )

        return bids
