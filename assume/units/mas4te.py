# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import logging

from battery_utility_calculator import Storage

from assume.common.base import BaseUnit
from assume.common.forecasts import Forecaster
from assume.common.market_objects import Order

try:  # ElectricVehicle is only available in newer BUC versions
    from battery_utility_calculator import ElectricVehicle
except ImportError:  # pragma: no cover - depends on installed BUC version
    ElectricVehicle = None

logger = logging.getLogger(__name__)
EPS = 1e-4

# static (non time series) parameters of a BUC ElectricVehicle that are
# exposed in as_dict() and in the market_open message to the AI agents
EV_STATIC_PARAMS = (
    "capacity_kwh",
    "max_charge_kw",
    "max_discharge_kw",
    "charge_efficiency",
    "discharge_efficiency",
    "initial_soc",
    "min_soc",
    "min_departure_soc",
    "public_charging_price",
)


def electric_vehicle_params(electric_vehicle) -> dict | None:
    """
    Return the static parameters of a BUC ``ElectricVehicle`` as a JSON
    serialisable dict, or None if there is no EV.

    The time series (``availability``, ``driving_consumption``) are left out on
    purpose, they are part of the forecaster of the unit.
    """
    if electric_vehicle is None:
        return None

    params = {"id": getattr(electric_vehicle, "id", None)}
    for name in EV_STATIC_PARAMS:
        params[name] = getattr(electric_vehicle, name, None)

    for key, value in params.items():
        if value is None:
            continue
        if key == "id":
            params[key] = value if isinstance(value, int | str) else str(value)
        else:
            params[key] = float(value)

    # V2H is only possible with a discharge power > 0
    params["v2h"] = bool(params["max_discharge_kw"] or 0)
    return params


class MAS4TEUnit(BaseUnit):
    """Class for a MAS4TE unit."""

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        baseline_storage: Storage,
        electric_vehicle: "ElectricVehicle | None" = None,
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

        self.baseline_storage = baseline_storage
        # optional BUC ElectricVehicle of the household, modelled as flexible load
        # in the BUC (not as a seller of storage), None if the unit has no EV
        self.electric_vehicle = electric_vehicle
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
                # JSON string (like str() for the storage) so it can be written
                # to the units_meta table, None if the unit has no EV
                "electric_vehicle": (
                    json.dumps(electric_vehicle_params(self.electric_vehicle))
                    if self.electric_vehicle is not None
                    else None
                ),
                "model_context": self.model_context,
                "previous_bids": self.previous_bids,
                "previous_accepted_bids": self.previous_accepted_bids,
                "previous_clearing_prices": self.previous_clearing_prices,
            }
        )

        return unit_dict
