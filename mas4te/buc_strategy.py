# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bidding with the Battery Utility Calculator (BUC), directly in the simulation.

The production system bids through the communications agents over MQTT
(``mas4te_bidding_strategy.py``). This strategy does the same calculation
in-process, so a simulation needs no broker, no agents, no LLM and no
forecasting service:

    bid       : BUC worth of every storage size -> (risk adjusted) bidding curve
                -> one order per kWh step, like ``communications_agent/src/market/bidding.py``
    after the clearing : the BUC schedule for the awarded volume -> flows of the
                household, handed to the ``Recorder``

The BUC sees the real profiles of the week (perfect foresight).
"""

import hashlib
import logging

import pandas as pd
from battery_utility_calculator import (
    EnergyCostCalculator,
    Storage,
    calculate_bidding_curve,
    calculate_multiple_storage_worth,
    calculate_multiple_storage_worth_by_location,
    calculate_risk_adjusted_bidding_curve,
    calculate_storage_worth_distribution,
    sample_scenarios,
)

from assume.common.base import BaseStrategy, BaseUnit
from assume.common.market_objects import MarketConfig, Orderbook
from grid_sim_io import EPS, Inputs, Recorder, UnitSpec, window

log = logging.getLogger(__name__)

RISK_TOLERANCE_LEVELS = {"Low": 0.05, "Medium": 0.35, "High": 0.50}


class BUCSettings:
    """Parameters of the BUC bidding, shared by all units of a simulation."""

    def __init__(
        self,
        solver: str = "appsi_highs",
        risk_bidding: bool = False,
        n_scenarios: int = 12,
        seed: int | None = 42,
        c_rate: float = 0.2,
        max_buy_volume_kwh: int = 10,
        goal: str = "max_cashflow",
    ):
        self.solver = solver
        self.risk_bidding = risk_bidding  # True: risk adjusted curve over scenarios, False: one deterministic curve
        self.n_scenarios = n_scenarios
        self.seed = seed  # None: different scenarios in every run
        # kW per kWh of the storages the BUC evaluates, the agent uses 0.2
        # (the market itself only allows c_rate 1, that is the field of the order)
        self.c_rate = c_rate
        self.max_buy_volume_kwh = max_buy_volume_kwh  # upper bound of the buyer's search range
        self.goal = goal  # "max_cashflow" or "max_green_energy"


class BUCStrategy(BaseStrategy):
    """In-process BUC bidding for one household (buyer or seller)."""

    def __init__(
        self,
        unit_id: str,
        spec: UnitSpec,
        inputs: Inputs,
        recorder: Recorder,
        settings: BUCSettings | None = None,
        market_config: MarketConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.unit_id = unit_id
        self.spec = spec
        self.inputs = inputs
        self.recorder = recorder
        self.settings = settings or BUCSettings()
        self.market_config = market_config
        self.bid_counter = 0
        self.last_series: dict | None = None
        # not a real address: the clearing only needs a non-empty string, there is no blockchain here
        self.blockchain_address = "0x" + format(int(hashlib.sha1(unit_id.encode()).hexdigest(), 16) % 16**40 + 10**6, "040x")

    # -- inputs of the BUC -------------------------------------------------
    def _series(self, start, end) -> dict:
        """The time series of the delivery week (perfect foresight), as the BUC wants them."""
        prices = self.inputs.prices
        return {
            "demand": window(self.spec.load_kw, start, end),
            "solar_generation": window(self.spec.solar_kw, start, end),
            "supplier_prices": window(prices["supplier"], start, end),
            "eeg_prices": window(prices["eeg"], start, end),
            "wholesale_market_prices": window(prices["wholesale"], start, end),
            "community": window(prices["community"], start, end),
        }

    def _locations_in_scope(self) -> list[str]:
        if self.spec.trading_scope == "Community":
            return [self.spec.location]
        return self.inputs.locations

    # -- bidding -----------------------------------------------------------
    def bidding_curve(self, series: dict) -> pd.DataFrame:
        """Worth of the storage sizes -> bidding curve, same calls as the communications agent."""
        spec, cfg = self.spec, self.settings
        h = self.inputs.step_hours
        is_buyer = spec.role == "buy"
        series = dict(series)
        community = series.pop("community")
        use_risk = cfg.risk_bidding and cfg.goal != "max_green_energy"
        my_location = spec.location

        if is_buyer:
            net_demand = (series["demand"] - series["solar_generation"]).clip(lower=0)
            max_volume = max(5, min(cfg.max_buy_volume_kwh, int(net_demand.max()) + 3))
            volumes = range(1, max_volume + 1)
            locations = self._locations_in_scope()
            baseline = Storage(id=0, c_rate=cfg.c_rate, volume=0)
            max_tradeable = None
        else:
            volumes = range(1, int(spec.battery_kwh) + 1)
            locations = [my_location]
            baseline = Storage(id=int(spec.battery_kwh), c_rate=cfg.c_rate, volume=spec.battery_kwh)
            max_tradeable = int(spec.battery_kwh * spec.tradeable_pct / 100)
        candidates = [Storage(id=v, c_rate=cfg.c_rate, volume=v) for v in volumes]

        if use_risk:
            scenarios = sample_scenarios(
                **series,
                community_market_prices={loc: community for loc in locations},
                n_scenarios=cfg.n_scenarios,
                seed=cfg.seed,
                hours_per_timestep=h,
            )
            rows = [
                calculate_storage_worth_distribution(
                    baseline_storage=baseline,
                    storages_to_calculate=candidates,
                    scenarios=scenarios,
                    hours_per_timestep=h,
                    my_location=my_location,
                    storage_location=loc if is_buyer else None,
                    is_rented_storage=is_buyer,
                    goal=cfg.goal,
                    solver=cfg.solver,
                )
                for loc in locations
            ]
            worth = pd.concat(rows, ignore_index=True)
            if is_buyer:  # the baseline row is only needed once
                worth = worth[
                    ~((worth["location"] != my_location) & (worth["volume"] == baseline.volume))
                ].reset_index(drop=True)
            curve = calculate_risk_adjusted_bidding_curve(
                worth_distribution=worth,
                buy_or_sell_side="buyer" if is_buyer else "seller",
                risk_level=RISK_TOLERANCE_LEVELS[spec.risk_tolerance],
            )
        else:
            if is_buyer:
                results = calculate_multiple_storage_worth_by_location(
                    baseline_storage=baseline,
                    storages_to_calculate=candidates,
                    locations_to_calculate=locations,
                    community_market_prices={loc: community for loc in locations},
                    my_location=my_location,
                    goal=cfg.goal,
                    solver=cfg.solver,
                    hours_per_timestep=h,
                    **series,
                )
            else:
                results = calculate_multiple_storage_worth(
                    baseline_storage=baseline,
                    storages_to_calculate=candidates,
                    community_market_prices={my_location: community},
                    my_location=my_location,
                    is_rented_storage=False,
                    goal=cfg.goal,
                    solver=cfg.solver,
                    hours_per_timestep=h,
                    **series,
                )
            columns = ["volume", "worth"] + (["location"] if "location" in results.columns else [])
            curve = calculate_bidding_curve(
                volumes_worth=results[columns],
                buy_or_sell_side="buyer" if is_buyer else "seller",
            )

        if max_tradeable is not None:
            curve = curve.head(max_tradeable)
        return curve

    def calculate_bids(
        self,
        unit: BaseUnit,
        market_config: MarketConfig,
        product_tuples: list[tuple],
        **kwargs,
    ) -> Orderbook:
        start, end, only_hours = product_tuples[0]  # one product per auction: the next week
        self.last_series = self._series(start, end)
        curve = self.bidding_curve(self.last_series)
        c_rate = market_config.param_dict["allowed_c_rates"][0]
        sign = 1 if self.spec.role == "sell" else -1

        week = self.bid_counter
        self.bid_counter += 1
        orders = []
        for i, row in curve.iterrows():
            location = row.get("location") or self.spec.location
            orders.append(
                {
                    # unique per week, unit and step, the clearing needs that
                    "bid_id": f"{unit.id}_w{week}_{location}_{i + 1}",
                    "start_time": start,
                    "end_time": end,
                    "only_hours": only_hours if only_hours is not None else [],
                    "volume": sign * float(row["volume"]),
                    "price": float(row["marginal_price_per_kwh"]),
                    "agent_addr": getattr(unit, "addr", "agent_addr_unknown"),
                    "node": getattr(unit, "node", "node0"),
                    "location": location,
                    "exclusive_id": row.get("exclusive_id"),
                    "c_rate": c_rate,
                    "blockchain_address": self.blockchain_address,
                }
            )
        self.recorder.register_bids(unit.id, [o["bid_id"] for o in orders])
        log.info("%s bids %d steps, %.0f kWh", unit.id, len(orders), sum(abs(o["volume"]) for o in orders))
        return orders

    # -- after the clearing ------------------------------------------------
    def calculate_reward(
        self, unit: BaseUnit, marketconfig: MarketConfig, orderbook: Orderbook
    ) -> None:
        """Called with the cleared orders of the unit: record them and the awarded schedule."""
        spec = self.spec
        orders = [dict(o) for o in orderbook]
        self.recorder.record_result(unit.id, spec.role, orders)

        accepted = [o for o in orders if abs(o.get("accepted_volume") or 0.0) > EPS]
        awarded_kwh = sum(abs(o["accepted_volume"]) for o in accepted)
        if not accepted and not self.recorder.export_unawarded:
            return

        # the location the buyer rents most storage at (the BUC evaluates one storage location)
        by_location: dict[str, float] = {}
        for o in accepted:
            by_location[o["location"]] = by_location.get(o["location"], 0.0) + abs(o["accepted_volume"])
        storage_location = max(by_location, key=by_location.get) if by_location else spec.location
        if spec.role == "buy" and len(by_location) > 1:
            log.warning("%s rents at several locations %s, schedule uses %s", unit.id, by_location, storage_location)

        start, end = orders[0]["start_time"], orders[0]["end_time"]
        # buyer: the rented volume is the storage; seller: the part that is not rented out
        volume = awarded_kwh if spec.role == "buy" else spec.battery_kwh - awarded_kwh
        flows, cashflows = self.household_flows(self.last_series, volume, storage_location)

        rent = sum(abs(o["accepted_volume"]) * o["accepted_price"] for o in accepted)
        self.recorder.record_week(
            {
                "unit_id": unit.id,
                "role": spec.role,
                "location": spec.location,
                "storage_location": storage_location,
                "product_start": start,
                "product_end": end,
                "awarded_kwh": awarded_kwh,
                "own_storage_kwh": volume if spec.role == "sell" else 0.0,
                # buyer pays, seller receives (EUR for the week)
                "rent_eur": -rent if spec.role == "buy" else rent,
                **{f"cashflow_{k}_eur": v for k, v in cashflows.items()},
                "flows": flows,
            }
        )

    def household_flows(
        self, series: dict, volume: float, storage_location: str
    ) -> tuple[pd.DataFrame, dict]:
        """Optimal schedule of the household for a storage of ``volume`` kWh.

        Returns the flows in kW and the cashflow components in EUR.
        """
        spec, cfg = self.spec, self.settings
        h = self.inputs.step_hours
        community = series["community"]
        is_buyer = spec.role == "buy"
        ecc = EnergyCostCalculator(
            storage=Storage(id=0, c_rate=cfg.c_rate, volume=volume) if volume > EPS else None,
            demand=series["demand"],
            solar_generation=series["solar_generation"],
            supplier_prices=series["supplier_prices"],
            eeg_prices=series["eeg_prices"],
            wholesale_market_prices=series["wholesale_market_prices"],
            community_market_prices={storage_location: community} if not is_buyer else {
                loc: community for loc in {spec.location, storage_location}
            },
            hours_per_timestep=h,
            is_rented_storage=is_buyer,
            my_location=spec.location,
            storage_location=storage_location if is_buyer else None,
            goal=cfg.goal,
        )
        ecc.optimize(solver=cfg.solver)
        cashflows = ecc.get_cashflows()
        kw = ecc.get_energy_flows().fillna(0.0) / h  # kWh per step -> kW
        kw = kw.mask(kw.abs() < 1e-9, 0.0)  # solver noise, no "-0.0"

        def col(name):
            return kw[name] if name in kw else 0.0

        flows = pd.DataFrame(index=kw.index)
        flows["demand_kw"] = series["demand"].to_numpy()
        flows["solar_kw"] = series["solar_generation"].to_numpy()
        # as seen at the meter of the household: what is drawn from / fed into the grid
        flows["grid_import_kw"] = (
            col("supplier_to_home") + col("supplier_to_storage") + col("wholesale_to_storage")
            + col("community_to_home") + col("community_to_storage")
        )
        flows["grid_export_kw"] = (
            col("pv_to_wholesale") + col("pv_to_eeg") + col("pv_to_community")
            + col("storage_to_wholesale") + col("storage_to_eeg") + col("storage_to_community")
        )
        flows["net_grid_kw"] = flows["grid_import_kw"] - flows["grid_export_kw"]
        flows["net_grid_without_storage_kw"] = flows["demand_kw"] - flows["solar_kw"]

        # the single flows, for those who want to recombine them
        for name in kw.columns:
            if name != "demand":
                flows[f"flow_{name}_kw"] = kw[name]

        if volume > EPS:
            soc = ecc.get_storage_soc_timeseries_df()
            soc = soc.mask(soc.abs() < 1e-9, 0.0)
            flows["storage_soc_kwh"] = soc.sum(axis=1).to_numpy()
            # SOC change per hour: positive = charging, negative = discharging (before conversion losses)
            flows["storage_charge_kw"] = (
                soc.sum(axis=1).diff().fillna(soc.sum(axis=1).iloc[0]) / h
            ).to_numpy()
            for use_case in soc.columns:
                flows[f"{use_case}_kwh"] = soc[use_case].to_numpy()
        else:
            flows["storage_soc_kwh"] = 0.0
            flows["storage_charge_kw"] = 0.0
        return flows, cashflows
