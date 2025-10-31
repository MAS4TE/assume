# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from multiprocessing import Process, Queue

import communication_agent
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
        self.storages_to_calculate = self.build_storages_to_calculate()

        self.market_to_llm_queue = Queue()
        self.llm_to_market_queue = Queue()

        self.process = Process(
            target=communication_agent.run_app,
            daemon=True,
            kwargs={
                "port": kwargs.get("comm_agent_port", 8000),
                "market_to_llm_queue": self.market_to_llm_queue,
                "llm_to_market_queue": self.llm_to_market_queue,
            },
        )
        self.process.start()

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

        self.market_to_llm_queue.put(
            {
                "msg": "calculate bids",
                # "market_config": market_config,
                "product_tuples": product_tuples[0],
            }
        )

        bids = self.llm_to_market_queue.get()

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

        self.market_to_llm_queue.put(
            {
                "msg": "market result",
                # "market_config": marketconfig,
                "orderbook": orderbook,
            }
        )


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

        self.market_to_llm_queue.put(
            {
                "msg": "calculate bids",
                # "market_config": market_config,
                "product_tuples": product_tuples[0],
            }
        )

        bids = self.llm_to_market_queue.get()

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

        self.market_to_llm_queue.put(
            {
                "msg": "market result",
                # "market_config": marketconfig,
                "orderbook": orderbook,
            }
        )
