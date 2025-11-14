# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os

import battery_utility_calculator as buc
import pandas as pd
from battery_utility_calculator import Storage
from dotenv import load_dotenv
from sqlalchemy.exc import ProgrammingError

from assume.common.base import BaseStrategy, BaseUnit, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product

load_dotenv()
logger = logging.getLogger(__name__)
DB_URI = os.getenv("DB_URI")


class LLMStrategy(BaseStrategy):
    """
    A strategy that uses a Large Language Model (LLM) for a storage buyer.

    Params:
        llm_api_url (str): The URL of the LLM API to use for generating bids.
    """

    def __init__(self, baseline_storage=0, *args, **kwargs):
        super().__init__()
        self.baseline_storage = baseline_storage

    def build_storages_to_calculate(
        self,
        baseline_storage: Storage,
        max_storage_volume: float = 5,
    ) -> list[Storage]:
        """Builds a list of storage volumes to calculate worth for.

        Returns:
            list[Storage]: List of Storage objects with different volumes.
        """
        storages = [
            Storage(
                id=i,
                volume=i / 2,
                c_rate=1,
                charge_efficiency=0.98,
                discharge_efficiency=0.98,
            )
            for i in range(max_storage_volume * 2 + 1)
        ]

        if baseline_storage.volume > 0:
            return [stor for stor in storages if stor.volume <= baseline_storage.volume]
        else:
            return storages

    def get_storages_worth_from_db(
        self,
        profile_id: int,
        product_start: str,
        product_end: str,
        hours_per_timestep: float | int,
        c_rate: float | int | None = None,
        charge_efficiency: float | int | None = None,
        discharge_efficiency: float | int | None = None,
    ) -> pd.DataFrame:
        sql = f"""
            SELECT
                volume,
                worth
            FROM
                storage_values.values
            WHERE
                product_start = '{product_start}'
            AND
                product_end = '{product_end}'
            AND
                hours_per_timestep = {hours_per_timestep}
            AND
                profile_id = {profile_id}
        """
        try:
            volume_values = pd.read_sql(sql, con=DB_URI)
        except ProgrammingError:
            return pd.DataFrame()

        volume_values.drop_duplicates(inplace=True)

        return volume_values

    def write_volumes_worth_to_db(
        self,
        profile_id: int,
        product_start: str,
        product_end: str,
        hours_per_timestep: float | int,
        storages_worth: pd.DataFrame,
    ) -> None:
        storages_worth["product_start"] = product_start
        storages_worth["product_end"] = product_end
        storages_worth["hours_per_timestep"] = hours_per_timestep
        storages_worth["profile_id"] = profile_id
        storages_worth["costs"] = storages_worth["costs"].astype(float)

        storages_worth.to_sql(
            name="values", con=DB_URI, schema="storage_values", if_exists="append"
        )


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
            baseline_storage=unit.baseline_storage,
            max_storage_volume=unit.maxbidding_volume,
        )
        max_vol_to_calc = max([s.volume for s in storages_to_calculate])

        # iterate over each product (which is only one in phase 1)
        for product in product_tuples:
            start = product[0]
            end = product[1]

            # get existing worths from DB
            existing_storages_worth = self.get_storages_worth_from_db(
                product_start=start,
                product_end=end,
                hours_per_timestep=0.25,
                profile_id=unit.profile_id,
            )

            # remove storages too large
            existing_storages_worth = existing_storages_worth[
                existing_storages_worth["volume"] <= max_vol_to_calc
            ].reset_index(drop=True)

            # find missing storage volumes that need to be calculated
            missing_storages = [
                s
                for s in storages_to_calculate
                if s.volume not in existing_storages_worth["volume"].values
            ]

            # calculate missing storage worths
            if len(missing_storages) > 0:
                new_storages_worth = buc.calculate_multiple_storage_worth(
                    baseline_storage=unit.baseline_storage,
                    storages_to_calculate=missing_storages,
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
                    community_market_prices=unit.forecaster[
                        "community_price"
                    ].as_pd_series(start=start, end=end),
                    wholesale_market_prices=unit.forecaster[
                        "wholesale_price"
                    ].as_pd_series(start=start, end=end),
                    solver="gurobi",
                    hours_per_timestep=0.25,
                )

                # write newly calculated worths to DB
                self.write_volumes_worth_to_db(
                    profile_id=unit.profile_id,
                    product_start=start,
                    product_end=end,
                    hours_per_timestep=0.25,
                    storages_worth=new_storages_worth,
                )

                # combine existing with new worths
                storages_worth = pd.concat(
                    [existing_storages_worth, new_storages_worth], ignore_index=True
                )
            else:
                storages_worth = existing_storages_worth.copy()

            # calculate bidding curve
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
                        "price": row["marginal_price_per_kwh"],
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
            baseline_storage=unit.baseline_storage,
            max_storage_volume=unit.max_bidding_volume,
        )
        max_vol_to_calc = max([s.volume for s in storages_to_calculate])

        # iterate over each product (which is only one in phase 1)
        for product in product_tuples:
            start = product[0]
            end = product[1]

            # get existing worths from DB
            existing_storages_worth = self.get_storages_worth_from_db(
                product_start=start,
                product_end=end,
                hours_per_timestep=0.25,
                profile_id=unit.profile_id,
            )

            # remove storages too large
            existing_storages_worth = existing_storages_worth[
                existing_storages_worth["volume"] <= max_vol_to_calc
            ].reset_index(drop=True)

            # find missing storage volumes that need to be calculated
            missing_storages = [
                s
                for s in storages_to_calculate
                if s.volume not in existing_storages_worth["volume"].values
            ]

            # calculate missing storage worths
            if len(missing_storages) > 0:
                new_storages_worth = buc.calculate_multiple_storage_worth(
                    baseline_storage=unit.baseline_storage,
                    storages_to_calculate=missing_storages,
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
                    community_market_prices=unit.forecaster[
                        "community_price"
                    ].as_pd_series(start=start, end=end),
                    wholesale_market_prices=unit.forecaster[
                        "wholesale_price"
                    ].as_pd_series(start=start, end=end),
                    solver="gurobi",
                    hours_per_timestep=0.25,
                )

                # write newly calculated worths to DB
                self.write_volumes_worth_to_db(
                    profile_id=unit.profile_id,
                    product_start=start,
                    product_end=end,
                    hours_per_timestep=0.25,
                    storages_worth=new_storages_worth,
                )

                # combine existing with new worths
                storages_worth = pd.concat(
                    [existing_storages_worth, new_storages_worth], ignore_index=True
                )
            else:
                storages_worth = existing_storages_worth.copy()

            # calculate bidding curve
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
                        "price": row["marginal_price_per_kwh"],
                        "volume": row["volume"],
                        "c_rate": 1,
                    }
                )

        return bids
