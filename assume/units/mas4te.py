# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

from battery_utility_calculator import Storage

from assume.common.base import BaseUnit
from assume.common.forecasts import Forecaster
from assume.common.market_objects import Order

logger = logging.getLogger(__name__)
EPS = 1e-4


class MAS4TEUnit(BaseUnit):
    """Class for a MAS4TE unit."""

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        profile_id: int,
        baseline_storage: Storage,
        max_bidding_volume: float | int = 5,
        model_context: list[str] | None = None,
        previous_bids: list[Order] | None = None,
        previous_accepted_bids: list[Order] | None = None,
        previous_clearing_prices: list[float] | None = None,
        location: tuple[float, float] = (0, 0),
        node: str = "node0",
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )

        self.profile_id = profile_id
        self.baseline_storage = baseline_storage
        self.max_bidding_volume = max_bidding_volume
        self.model_context = model_context
        self.previous_bids = previous_bids
        self.previous_accepted_bids = previous_accepted_bids
        self.previous_clearing_prices = previous_clearing_prices

    def as_dict(self) -> dict:
        """
        Return the storage unit's attributes as a dictionary, including specific attributes.

        Returns:
            dict: The storage unit's attributes as a dictionary.
        """
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "baseline_storage": str(self.baseline_storage),
                "model_context": self.model_context,
                "previous_bids": self.previous_bids,
                "previous_accepted_bids": self.previous_accepted_bids,
                "previous_clearing_prices": self.previous_clearing_prices,
            }
        )

        return unit_dict
