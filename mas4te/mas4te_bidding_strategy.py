from assume.common.base import BaseStrategy, SupportsMinMax, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
import requests

class LLMStrategy(BaseStrategy):
    """
    A simple strategy where the user can input the bids of an agent manually.
    Only one agent should have this strategy in a simulation in one process, so that the terminal does not wait for multiple responses.

    Methods
    -------
    """
    def __init__(self, api_url="http://localhost:1234/v1/completions", *args, **kwargs ):
        super().__init__()
        self.api_url = api_url
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
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        start = product_tuples[0][0]  # start time of the first product
        end_all = product_tuples[-1][1]  # end time of the last product
        previous_power = unit.get_output_before(
            start
        )  # power output of the unit before the start time of the first product
        op_time = unit.get_operation_time(start)
        min_power_values, max_power_values = unit.calculate_min_max_power(
            start, end_all
        )  # minimum and maximum power output of the unit between the start time of the first product and the end time of the last product

        bids = []

        # TODO create only a single llm query
        for product, min_power, max_power in zip(
            product_tuples, min_power_values, max_power_values
        ):
            # for each product, calculate the marginal cost of the unit at the start time of the product
            # and the volume of the product. Dispatch the order to the market.
            start = product[0]
            current_power = unit.outputs["energy"].at[
                start
            ]  # power output of the unit at the start time of the current product
            marginal_cost = unit.calculate_marginal_cost(
                start, previous_power
            )  # calculation of the marginal costs
            volume = unit.calculate_ramp(
                op_time, previous_power, max_power, current_power
            )
            print(f"> requesting bid for time from {product[0]} to {product[1]}")
            print(f"> power must be between {min_power} and {max_power}")
            print(f"> {previous_power=}, {current_power=}, {marginal_cost=}")
            try:
                # prompt = input(
                #     "> waiting for volume and price, space-separated with a dot as decimal point \n"
                # )
                # volume, price = prompt.split(" ")
                # volume, price = float(volume), float(price)
                #price_forecast = unit.forecaster.get_price(unit.fuel_type)
                price_forecast = unit.forecaster.get_price("demand")
                price = 0
                volume = 0

                # prompt = "create prices and volumes for following products: {product_tuples} with the forecast: {price_forecast}"

                # TODO bea get response from LLM
                # volume, price = run_prompt( ... )
            except ValueError:
                volume, price = 0, 0

            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": price,
                    "volume": volume,
                    "node": unit.node,
                }
            )

            if "node" in market_config.additional_fields:
                bids[-1]["max_power"] = unit.max_power if volume > 0 else unit.min_power
                bids[-1]["min_power"] = (
                    min_power[start] if volume > 0 else unit.max_power
                )

            previous_power = volume + current_power
            if previous_power > 0:
                op_time = max(op_time, 0) + 1
            else:
                op_time = min(op_time, 0) - 1

        if "node" in market_config.additional_fields:
            return bids
        else:
            return self.remove_empty_bids(bids)
