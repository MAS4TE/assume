import pyomo.environ as pyo
import pandas as pd

from itertools import product
from datetime import date
import datetime
import logging

log = logging.getLogger("optimization")
log.setLevel(logging.INFO)


class Storage:
    def __init__(self, id: int, c_rate: float, volume: float, efficiency: float):
        """Represents a storage unit for energy.

        Args:
            id (int): The unique identifier for the storage unit.
            c_rate (float): The charging rate of the storage unit (kWh/h).
            volume (float): The total capacity of the storage unit (kWh).
            efficiency (float): The efficiency of the storage unit (0-1).
        """
        self.id = id
        self.c_rate = c_rate
        self.volume = volume
        self.efficiency = efficiency


class Optimizer:
    def __init__(
        self,
        storages: list[Storage],
        start_date: date,
        end_date: date,
        prices: pd.DataFrame,
        solar_generation: pd.Series,
        demand: pd.Series,
        storage_use_cases: list[str] = ["eeg", "wholesale", "community", "home"],
    ):
        """Optimizer for prosumer energy management, giving price recommendations for given products and given timeframes

        Args:
            storages (list[Storage]): The available storages
            start_date (date): The start date for the optimization. Should be in index of prices and solar_generation.
            end_date (date): The end date for the optimization. Should be in index of prices and solar_generation.
            demand (pd.Series): The demand data for the optimization. Values should be in kWh per hour (kW).
            prices (pd.DataFrame): The price data for the optimization. Columns should be "eeg", "wholesale", "community", "grid" with values in €/kWh.
            solar_generation (pd.Series): The solar generation data for the optimization. Values should be in kWh per hour (kW).
            storage_use_cases (list[str]): The use cases for energy storage. Allowed values are "eeg", "wholesale", "community", "home"
        """

        self.prices = prices
        self.solar_generation = solar_generation
        self.demand = demand
        self.storage_use_cases = storage_use_cases

        self.__check_timeseries_indices__()

        self.start_date = start_date
        self.end_date = end_date
        self.timesteps = pd.date_range(
            start=start_date, end=end_date, freq="h", tz="UTC"
        )

        self.storage_ids = [storage.id for storage in storages]
        self.storages = {storage.id: storage for storage in storages}

        self.prices = prices
        self.solar_generation = solar_generation
        self.demand = demand

        self.is_optimized = False

        self.model = pyo.ConcreteModel()
        self.set_model_variables()
        self.set_model_constraints()
        self.set_model_objective()

    def __check_timeseries_indices__(self):
        """Check if the indices of the timeseries are valid."""

        # check if indices are datetime indices
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise ValueError("Prices index must be a DatetimeIndex.")
        if not isinstance(self.solar_generation.index, pd.DatetimeIndex):
            raise ValueError("Solar generation index must be a DatetimeIndex.")
        if not isinstance(self.demand.index, pd.DatetimeIndex):
            raise ValueError("Demand index must be a DatetimeIndex.")

        # check if indices are in UTC timezone
        if self.prices.index.tz != datetime.timezone.utc:
            raise ValueError("Prices index must be in UTC timezone.")
        if self.solar_generation.index.tz != datetime.timezone.utc:
            raise ValueError("Solar generation index must be in UTC timezone.")
        if self.demand.index.tz != datetime.timezone.utc:
            raise ValueError("Demand index must be in UTC timezone.")

        # check if indices are equal
        if not self.prices.index.equals(self.solar_generation.index):
            raise ValueError("Prices and solar generation indices must be equal.")
        if not self.prices.index.equals(self.demand.index):
            raise ValueError("Prices and demand indices must be equal.")
        if not self.solar_generation.index.equals(self.demand.index):
            raise ValueError("Solar generation and demand indices must be equal.")

    def set_model_variables(self):
        log.info("Setting up model variables...")

        # Selling PV for EEG
        self.model.pv_to_eeg = pyo.Var(self.timesteps, domain=pyo.NonNegativeReals)

        # Selling PV on wholesale
        self.model.pv_to_wholesale = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals
        )

        # Selling PV on community market
        # remove bounds for activation
        self.model.pv_to_community = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
        )

        # PV energy to storage
        self.model.pv_to_storage = pyo.Var(
            self.timesteps,
            self.storage_use_cases,
            self.storage_ids,
            domain=pyo.NonNegativeReals,
        )

        # using PV energy at home
        self.model.pv_to_home = pyo.Var(self.timesteps, domain=pyo.NonNegativeReals)

        # storage level restrction
        self.model.storage_level = pyo.Var(
            self.timesteps,
            self.storage_use_cases,
            self.storage_ids,
            domain=pyo.NonNegativeReals,
        )

        # selling from storage for EEG
        self.model.storage_to_eeg = pyo.Var(
            self.timesteps, self.storage_ids, domain=pyo.NonNegativeReals
        )

        # seeling from storage on wholesale
        self.model.storage_to_wholesale = pyo.Var(
            self.timesteps, self.storage_ids, domain=pyo.NonNegativeReals
        )

        # selling from storage on community
        self.model.storage_to_community = pyo.Var(
            self.timesteps,
            self.storage_ids,
            domain=pyo.NonNegativeReals,
            bounds=(0, 0),
        )

        # using energy from storage at home
        self.model.storage_to_home = pyo.Var(
            self.timesteps, self.storage_ids, domain=pyo.NonNegativeReals
        )

        # charging storage from wholesale market
        self.model.wholesale_to_storage = pyo.Var(
            self.timesteps, self.storage_ids, domain=pyo.NonNegativeReals
        )

        # charging storage from community market
        # remove bounds for activation
        self.model.community_to_storage = pyo.Var(
            self.timesteps,
            self.storage_ids,
            domain=pyo.NonNegativeReals,
            bounds=(0, 0),
        )

        # buying energy from community market
        # remove bounds for activation
        self.model.community_to_home = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
        )

        # charging storage from supplier
        self.model.supplier_to_storage = pyo.Var(
            self.timesteps, self.storage_ids, domain=pyo.NonNegativeReals
        )

        # buying energy from supplier
        self.model.supplier_to_home = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals
        )

        log.info("Model variables set up successfully.")

    def set_model_constraints(self):
        log.info("Setting up model constraints...")

        # consumption must equal supply (from PV system, supplier, community market, PV system)
        def restrict_demand(model, timestep):
            return (
                sum(model.storage_to_home[timestep, id] for id in self.storage_ids)
                + model.pv_to_home[timestep]
                + model.supplier_to_home[timestep]
                + model.community_to_home[timestep]
                == self.demand.loc[timestep]
            )

        self.model.demand_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_demand
        )

        # use of PV system must be smaller or equal than PV system generation
        def restrict_solar_gen(model, timestep):
            uses_ids = list(product(self.storage_use_cases, self.storage_ids))
            return (
                sum(model.pv_to_storage[timestep, use, id] for use, id in uses_ids)
                + model.pv_to_eeg[timestep]
                + model.pv_to_home[timestep]
                + model.pv_to_wholesale[timestep]
                + model.pv_to_community[timestep]
                <= self.solar_generation.loc[timestep]
            )

        self.model.solar_gen_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_solar_gen
        )

        # energy flow TO storage must be smaller than c_rate
        def restrict_storage_charge(model, timestep, storage_id):
            return (
                sum(
                    model.pv_to_storage[timestep, use, storage_id]
                    for use in self.storage_use_cases
                )
                + model.wholesale_to_storage[timestep, storage_id]
                + model.community_to_storage[timestep, storage_id]
                + model.supplier_to_storage[timestep, storage_id]
                <= self.storages[storage_id].c_rate
            )

        self.model.storage_charge_restriction = pyo.Constraint(
            self.timesteps, self.storage_ids, rule=restrict_storage_charge
        )

        # energy flow FROM storage must be smaller than c_rate
        def restrict_storage_discharge(model, timestep, storage_id):
            return (
                +model.storage_to_eeg[timestep, storage_id]
                + model.storage_to_wholesale[timestep, storage_id]
                + model.storage_to_community[timestep, storage_id]
                + model.storage_to_home[timestep, storage_id]
                <= self.storages[storage_id].c_rate
            )

        self.model.storage_discharge_restriction = pyo.Constraint(
            self.timesteps, self.storage_ids, rule=restrict_storage_discharge
        )

        # storage level must be smaller than volume
        def restrict_soc_max(model, timestep, storage_id):
            return (
                sum(
                    model.storage_level[timestep, use, storage_id]
                    for use in self.storage_use_cases
                )
                <= self.storages[storage_id].volume
            )

        self.model.storage_level_restriction = pyo.Constraint(
            self.timesteps, self.storage_ids, rule=restrict_soc_max
        )

        # storage level must be larger than 0
        def restrict_soc_min(model, timestep, storage_id):
            return (
                sum(
                    model.storage_level[timestep, use, storage_id]
                    for use in self.storage_use_cases
                )
                >= 0
            )

        self.model.storage_level_non_negative = pyo.Constraint(
            self.timesteps, self.storage_ids, rule=restrict_soc_min
        )

        if "eeg" in self.storage_use_cases:

            def restrict_soc_eeg(model, timestep, storage_id):
                if timestep == self.timesteps[0]:
                    return (
                        model.storage_level[timestep, "eeg", storage_id]
                        == model.pv_to_storage[timestep, "eeg", storage_id]
                        - model.storage_to_eeg[timestep, storage_id]
                    )
                else:
                    previous_timestep = self.timesteps[
                        self.timesteps.get_loc(timestep) - 1
                    ]
                    return (
                        model.storage_level[timestep, "eeg", storage_id]
                        == model.storage_level[previous_timestep, "eeg", storage_id]
                        + model.pv_to_storage[timestep, "eeg", storage_id]
                        - model.storage_to_eeg[timestep, storage_id]
                    )

            self.model.soc_eeg_restriction = pyo.Constraint(
                self.timesteps, self.storage_ids, rule=restrict_soc_eeg
            )

        if "wholesale" in self.storage_use_cases:

            def restrict_soc_wholesale(model, timestep, storage_id):
                if timestep == self.timesteps[0]:
                    return (
                        model.storage_level[timestep, "wholesale", storage_id]
                        == model.wholesale_to_storage[timestep, storage_id]
                        - model.storage_to_wholesale[timestep, storage_id]
                    )
                else:
                    previous_timestep = self.timesteps[
                        self.timesteps.get_loc(timestep) - 1
                    ]
                    return (
                        model.storage_level[timestep, "wholesale", storage_id]
                        == model.storage_level[
                            previous_timestep, "wholesale", storage_id
                        ]
                        + model.wholesale_to_storage[timestep, storage_id]
                        - model.storage_to_wholesale[timestep, storage_id]
                    )

            self.model.soc_wholesale_restriction = pyo.Constraint(
                self.timesteps, self.storage_ids, rule=restrict_soc_wholesale
            )

        if "community" in self.storage_use_cases:

            def restrict_soc_community(model, timestep, storage_id):
                if timestep == self.timesteps[0]:
                    return (
                        model.storage_level[timestep, "community", storage_id]
                        == model.community_to_storage[timestep, storage_id]
                        - model.storage_to_community[timestep, storage_id]
                    )
                else:
                    previous_timestep = self.timesteps[
                        self.timesteps.get_loc(timestep) - 1
                    ]
                    return (
                        model.storage_level[timestep, "community", storage_id]
                        == model.storage_level[
                            previous_timestep, "community", storage_id
                        ]
                        + model.community_to_storage[timestep, storage_id]
                        - model.storage_to_community[timestep, storage_id]
                    )

            self.model.soc_community_restriction = pyo.Constraint(
                self.timesteps, self.storage_ids, rule=restrict_soc_community
            )

        if "home" in self.storage_use_cases:

            def restrict_soc_home(model, timestep, store):
                if timestep == self.timesteps[0]:
                    return (
                        model.storage_level[timestep, "home", store]
                        == model.supplier_to_storage[timestep, store]
                        - model.storage_to_home[timestep, store]
                    )
                else:
                    previous_timestep = self.timesteps[
                        self.timesteps.get_loc(timestep) - 1
                    ]
                    return (
                        model.storage_level[timestep, "home", store]
                        == model.storage_level[previous_timestep, "home", store]
                        + model.supplier_to_storage[timestep, store]
                        - model.storage_to_home[timestep, store]
                    )

            self.model.soc_home_restriction = pyo.Constraint(
                self.timesteps, self.storage_ids, rule=restrict_soc_home
            )

        log.info("Model constraints set up successfully.")

    def set_model_objective(self):
        log.info("Setting up model objective...")

        # community market cashflow
        community_cf = (
            # selling from storage to community market
            sum(
                self.model.storage_to_community[timestep, storage_id]
                * self.prices.loc[timestep, "community"]
                for timestep, storage_id in product(self.timesteps, self.storage_ids)
            )
            # selling from pv to community market
            + sum(
                self.model.pv_to_community[timestep]
                * self.prices.loc[timestep, "community"]
                for timestep in self.timesteps
            )
            # buying from community market to storage
            - sum(
                self.model.community_to_storage[timestep, storage_id]
                * self.prices.loc[timestep, "community"]
                for timestep, storage_id in product(self.timesteps, self.storage_ids)
            )
            # buying from community market to home
            - sum(
                self.model.community_to_home[timestep]
                * self.prices.loc[timestep, "community"]
                for timestep in self.timesteps
            )
        )

        # supplier cashflow
        supplier_cf = (
            # buying energy from supplier to storage
            -sum(
                self.model.supplier_to_storage[timestep, storage_id]
                * self.prices.loc[timestep, "grid"]
                for timestep, storage_id in product(self.timesteps, self.storage_ids)
            )
            # buying energy from supplier to home
            - sum(
                self.model.supplier_to_home[timestep]
                * self.prices.loc[timestep, "grid"]
                for timestep in self.timesteps
            )
        )

        # EEG cashflow
        eeg_cf = (
            # selling from storage for EEG
            sum(
                self.model.storage_to_eeg[timestep, storage_id]
                * self.prices.loc[timestep, "eeg"]
                for timestep, storage_id in product(self.timesteps, self.storage_ids)
            )
            # selling from PV for EEG
            + sum(
                self.model.pv_to_eeg[timestep] * self.prices.loc[timestep, "eeg"]
                for timestep in self.timesteps
            )
        )

        # wholesale cashflow
        wholesale_cf = (
            # selling from storage to wholesale
            sum(
                self.model.storage_to_wholesale[timestep, storage_id]
                * self.prices.loc[timestep, "wholesale"]
                for timestep, storage_id in product(self.timesteps, self.storage_ids)
            )
            # buying from wholesale to storage
            - sum(
                self.model.wholesale_to_storage[timestep, storage_id]
                * self.prices.loc[timestep, "wholesale"]
                for timestep, storage_id in product(self.timesteps, self.storage_ids)
            )
        )

        # maximize sum of cashflows
        self.model.objective = pyo.Objective(
            expr=community_cf + supplier_cf + eeg_cf + wholesale_cf, sense=pyo.maximize
        )

        log.info("Model objective set up successfully.")

    def optimize(self, solver: str = "gurobi"):
        optimizer = pyo.SolverFactory(
            solver,
        )

        results = optimizer.solve(self.model, tee=False)

        self.is_optimized = True

        return results

    def __build_demand_timeseries_df__(self):
        demand_coverage = pd.DataFrame(index=self.timesteps)

        demand_coverage["demand"] = self.demand.copy()
        demand_coverage["from_grid"] = [
            self.model.supplier_to_home[timestep].value for timestep in self.timesteps
        ]
        demand_coverage["from_pv"] = [
            self.model.pv_to_home[timestep].value for timestep in self.timesteps
        ]
        demand_coverage["from_storage"] = [
            sum(
                self.model.storage_to_home[timestep, storage_id].value
                for storage_id in self.storage_ids
            )
            for timestep in self.timesteps
        ]
        demand_coverage["from_community"] = [
            self.model.community_to_home[timestep].value for timestep in self.timesteps
        ]

        return demand_coverage

    def __build_pv_timeseries_df__(self):
        pv_usage = pd.DataFrame(index=self.timesteps)

        pv_usage["generation"] = self.solar_generation.loc[self.timesteps]

        pv_usage["to_home"] = [self.model.pv_to_home[t].value for t in self.timesteps]
        pv_usage["to_eeg"] = [self.model.pv_to_eeg[t].value for t in self.timesteps]
        pv_usage["to_community"] = [
            self.model.pv_to_community[t].value for t in self.timesteps
        ]
        pv_usage["to_wholesale"] = [
            self.model.pv_to_wholesale[t].value for t in self.timesteps
        ]
        pv_usage["to_storage_home"] = [
            sum(
                self.model.pv_to_storage[t, "home", sid].value
                for sid in self.storage_ids
            )
            for t in self.timesteps
        ]
        pv_usage["to_storage_eeg"] = [
            sum(
                self.model.pv_to_storage[t, "eeg", sid].value
                for sid in self.storage_ids
            )
            for t in self.timesteps
        ]
        pv_usage["to_storage_wholesale"] = [
            sum(
                self.model.pv_to_storage[t, "wholesale", sid].value
                for sid in self.storage_ids
            )
            for t in self.timesteps
        ]
        pv_usage["to_storage_community"] = [
            sum(
                self.model.pv_to_storage[t, "community", sid].value
                for sid in self.storage_ids
            )
            for t in self.timesteps
        ]

        return pv_usage

    def __build_storage_timeseries_df__(self):
        storage_usage = pd.DataFrame(index=self.timesteps)

        for use_case in self.storage_use_cases:
            storage_usage[f"soc_{use_case}"] = [
                self.model.storage_level[t, use_case, sid].value
                for t, sid in product(self.timesteps, self.storage_ids)
            ]

        return storage_usage

    def output_results(self):
        if not self.is_optimized:
            raise ValueError(
                "Model not optimized yet - run class method optimize() first!"
            )

        demand_timeseries = self.__build_demand_timeseries_df__()
        pv_timeseries = self.__build_pv_timeseries_df__()
        storage_timeseries = self.__build_storage_timeseries_df__()

        return {
            "demand": demand_timeseries,
            "pv": pv_timeseries,
            "storage": storage_timeseries,
        }
