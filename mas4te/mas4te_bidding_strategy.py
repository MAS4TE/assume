# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import requests
from pricing_framework import PricingFramework, Storage
from llm_model_api import LLMModelAPI

from assume.common.base import BaseStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
# from assume.common.utils import get_supported_solver

import re
import ast
import json

class LLMStrategy(BaseStrategy):
    """
    A strategy that uses a Large Language Model (LLM) for a storage buyer.

    Params:
        baseline_storage
    """

    def __init__(self, baseline_storage=0, *args, **kwargs):
        super().__init__()
        self.baseline_storage = baseline_storage
        self.llm_model = LLMModelAPI()

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
        #pricer.optimize(solver=get_supported_solver("gurobi"))
        pricer.optimize("appsi_highs")
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
            # pricer.optimize(solver=get_supported_solver("gurobi"))
            pricer.optimize("appsi_highs")

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

    def run_sell_prompt(self, prompt: str, optimization_framework: str):
        print(optimization_framework)
        prompt = (
        f"You are an energy trading agent.\n"
        f"You receive an optimization framework output for your predicted situation: {optimization_framework}. \n" 
        f"You have to decide a volume and a corresponding price to sell. \n"
        f"The optimization framework is computed based on your baseline prediction with your storage. \n"
        f"Return your decision as a JSON object with the following keys:\n"
        f'- "volume": an integer\n'
        f'- "price": a float\n'
        f'- "reasoning": a short string explanation\n'
        f"Do not include any other text outside of the json. No explanations, no headers.\n"
        )

        result_text = self.llm_model.query(prompt)
        vol, pr, r = self.parse_llm_json_output(result_text)
        return vol, pr, r
        # return result.get("choices", [{}])[0].get("text", "")

    def run_buy_prompt(self, prompt: str):
        prompt = (
        "You are an energy trading agent.\n"
        "Based on the market, a good price is €10 per volume. \n" 
        "You have a volume of 4 to buy. \n"
        "Return your trading decision in the format:\n"
        "('volume','price', 'reasoning')\n"
        "Do not include any other text. No explanations, no headers, just the triple.\n"
        )

        result_text = self.llm_model.query(prompt=prompt)
        vol, pr, r = self.check_output_format(result_text)
        return vol, pr, r
        # return result.get("choices", [{}])[0].get("text", "")


    def parse_llm_json_output(self, text):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                volume = int(data.get("volume", 0))
                price = float(data.get("price", 0))
                reasoning = str(data.get("reasoning", ""))
                return volume, price, reasoning
            except Exception as e:
                pass
        return 0, 0, "invalid"

    
    def check_output_format(self, text):
        text = text.strip()
        print(text)
        # Extract first match of anything inside a pair of parentheses
        try:
            # Extract the first valid-looking tuple using regex
            match = re.search(r'\(.*?\)', text, re.DOTALL)
            if not match:
                print('all zeros')
                return 0, 0, 0

            tuple_str = match.group(0)
            parsed = ast.literal_eval(tuple_str)

            if isinstance(parsed, tuple) and len(parsed) == 3:
                return parsed[0], parsed[1], parsed[2]
        except Exception as e:
            print(f'Parsing error: {e}')

        print('all zeros')
        return 0, 0, 0


    # def reformat_output_format(self, raw_text):
    #     # First try to parse raw_text directly
    #     vol, pr, r = self.check_output_format(raw_text)
    #     if (vol, pr, r) != (0, 0, 0):
    #         return vol, pr, r

    #     # If invalid, call LLM once to reformat
    #     model = "Mistral-7B-Instruct-v0.3-Q4_K_M"
    #     prompt = f"Reformat the following text into the format (volume, price, reasoning):\n\n{raw_text}"
    #     data = {"model": model, "prompt": prompt, "max_tokens": 1000}
    #     response = requests.post(self.api_url, headers=self.headers, json=data)
    #     result = response.json()
    #     llm_output = result.get("choices", [{}])[0].get("text", "").strip()

    #     # Check the LLM output
    #     return self.check_output_format(llm_output)


class LLMBuyStrategy(LLMStrategy):
    """A strategy that uses a Large Language Model (LLM) for a storage buyer."""

    def __init__(self, baseline_storage=0, *args, **kwargs):
        super().__init__()
        self.baseline_storage = baseline_storage
        self.llm_model = LLMModelAPI()

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

        llm_price_recommendation, llm_volume_recommendation, reasoning_output = self.run_buy_prompt('')
        print(llm_price_recommendation, llm_volume_recommendation)

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

    # def __init__(self, llm_api_url='http://localhost:3000/api/chat/completions', baseline_storage=0, *args, **kwargs):
    def __init__(self, llm_api_url='http://localhost:1234/v1/completions', baseline_storage=0, *args, **kwargs):
        super().__init__()
        self.baseline_storage = baseline_storage
        self.api_url = llm_api_url
        # self.token = 'sk-fb471a2a5dac4c01935f4690c045d1dd'
        # self.headers = {"Authorization": f'Bearer {self.token}',"Content-Type": "application/json"}
        self.headers = {"Content-Type": "application/json"}
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
        print("Input something to continue... [ Sell ]")
        input()



        #####################################################
        #               LLM CALL GOES HERE                  #
        #   TO CHOOSE FROM RECOMMENDATIONS OR ALTER THEM    #
        #####################################################
        llm_price_recommendation = 1
        llm_volume_recommendation = 1

        optimization_seller_data, optimization_sell_text = self.get_seller_volumes_with_text(volumes_values)

        llm_price_recommendation, llm_volume_recommendation, reasoning_output = self.run_sell_prompt('', optimization_sell_text)
        print('before bid ', llm_price_recommendation, llm_volume_recommendation)
        print('before bid ', type(llm_price_recommendation), type(llm_volume_recommendation))

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

    def get_seller_volumes_with_text(self, volumes_values):
        result = []
        lines = ["Volumes and their values for seller:"]
        
        for volume, value in volumes_values.items():
            if volume > self.baseline_storage:
                continue
            sell_volume = self.baseline_storage - volume
            marginal_worth = -value
            result.append((sell_volume, marginal_worth))
            lines.append(f"Volume selling: {sell_volume}, Marginal worth (min. to receive): {marginal_worth}")
        
        text_output = "\n".join(lines)
        return result, text_output

