# SPDX-FileCopyrightText: MAS$TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import random
from datetime import datetime, timedelta

import pandas as pd
from battery_market_clearing import BatteryClearing
from battery_market_strategy import BatteryMarketStrategy
from battery_utility_calculator import Storage
from community_eom_strategy import CommunityEOMStrategy
from dateutil import rrule as rr
from dotenv import load_dotenv

try:  # ElectricVehicle is only available in newer BUC versions
    from battery_utility_calculator import ElectricVehicle
except ImportError:
    ElectricVehicle = None

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.markets.clearing_algorithms import PayAsClearRole

log = logging.getLogger(__name__)

# community energy-only market (CommunityEOM) next to the battery market,
# off by default; True adds the market and lets every household bid on it
ENABLE_COMMUNITY_EOM = False

# Static parameters per vehicle type (types follow synload-de: tesla, vw_eup, none).
# capacity_kwh = usable battery capacity, max_charge_kw = AC charging power at the
# home wallbox, max_discharge_kw > 0 would enable V2H (car -> home only).
EV_TYPES = {
    "tesla": {"capacity_kwh": 57.5, "max_charge_kw": 11.0, "max_discharge_kw": 0.0},
    "vw_eup": {"capacity_kwh": 32.3, "max_charge_kw": 7.2, "max_discharge_kw": 0.0},
}


def read_ev_profile(index, id: str = "0", data_dir: str = "./example_data"):
    """Reads the EV time series of profile ``id`` from ``ev.csv`` and aligns them to ``index``.

    ``ev.csv`` has one pair of columns per profile: ``ev_availability_<id>``
    (1 = at home and plugged in, 0 = away) and ``ev_driving_consumption_kw_<id>``
    (kW drawn from the battery while driving). A file with the unsuffixed columns
    ``ev_availability`` / ``ev_driving_consumption_kw`` (one vehicle per file, see
    the EV data contract) is accepted as well.

    If the file covers the whole ``index`` it is used as is, otherwise (e.g. the
    one-week placeholder profile) it is repeated by weekday and time of day.

    Args:
        index (FastIndex | pd.DatetimeIndex): The index of the simulation.
        id (str, optional): The profile ID. Defaults to "0".
        data_dir (str, optional): Folder containing ``ev.csv``.

    Returns:
        dict: ``ev_availability`` and ``ev_driving_consumption_kw`` as pd.Series on ``index``.
    """
    if hasattr(index, "as_datetimeindex"):
        index = index.as_datetimeindex()

    profiles = pd.read_csv(
        f"{data_dir}/ev.csv", index_col=0, parse_dates=True, comment="#"
    )
    result = {}
    for name in ("ev_availability", "ev_driving_consumption_kw"):
        column = f"{name}_{id}" if f"{name}_{id}" in profiles else name
        profile = profiles[column]
        if profile.index[0] <= index[0] and profile.index[-1] >= index[-1]:
            series = profile.reindex(index)
        else:
            # repeat the profile week by weekday and time of day
            week = profile.groupby(
                [profile.index.day_of_week, profile.index.hour, profile.index.minute]
            ).first()
            keys = list(zip(index.day_of_week, index.hour, index.minute))
            series = pd.Series(week.reindex(keys).to_numpy(), index=index)
        if series.isna().any():
            raise ValueError(f"EV profile {column} does not cover the simulation index")
        result[name] = series.astype(float).rename(name)

    return result


def create_electric_vehicle(id, ev_type: str, ev_profile: dict, **kwargs):
    """Creates the BUC ElectricVehicle of a household from its type and profile.

    Args:
        id: The ID of the vehicle.
        ev_type (str): Key of ``EV_TYPES``.
        ev_profile (dict): Output of ``read_ev_profile``.
        **kwargs: Further ElectricVehicle parameters (e.g. ``min_departure_soc``).

    Returns:
        ElectricVehicle: The vehicle, to be passed as ``electric_vehicle`` to the unit.
    """
    if ElectricVehicle is None:
        raise ImportError(
            "battery_utility_calculator has no ElectricVehicle, "
            "install a BUC version with EV support or set n_ev_households = 0"
        )
    return ElectricVehicle(
        id=id,
        availability=ev_profile["ev_availability"],
        driving_consumption=ev_profile["ev_driving_consumption_kw"],
        **EV_TYPES[ev_type],
        **kwargs,
    )


def read_forecasts(start, end, id: int = 0, randomize: bool = False):
    """Reads the forecasts for a specific time period and unit ID.

    Args:
        start (datetime): The start time of the forecast period.
        end (datetime): The end time of the forecast period.
        id (int, optional): The ID of the unit to read forecasts for. Defaults to 0.
        randomize (bool, optional): Whether to randomize the forecasts. Defaults to False. Overwrites the ID.

    Returns:
        dict: A dictionary containing the forecasts for the specified time period and unit ID.
    """
    if randomize:
        val = random.randint(0, 29)
        id = "0" + str(val) if val < 10 else str(val)

    demand_forecast = pd.read_csv(
        "./example_data/demand.csv", index_col=0, parse_dates=True
    )["demand" + "_" + id][start:end]
    wholesale_price = pd.read_csv(
        "./example_data/prices.csv", index_col=0, parse_dates=True
    )["wholesale"][start:end]
    eeg_price = pd.read_csv("./example_data/prices.csv", index_col=0, parse_dates=True)[
        "eeg"
    ][start:end]
    community_price = pd.read_csv(
        "./example_data/prices.csv", index_col=0, parse_dates=True
    )["community"][start:end]
    grid_price = pd.read_csv(
        "./example_data/prices.csv", index_col=0, parse_dates=True
    )["grid"][start:end]
    solar_gen = pd.read_csv("./example_data/solar.csv", index_col=0, parse_dates=True)[
        "solar" + "_" + id
    ][start:end]

    return {
        "demand": demand_forecast,
        "wholesale_price": wholesale_price,
        "eeg_price": eeg_price,
        "community_price": community_price,
        "grid_price": grid_price,
        "solar_gen": solar_gen,
    }


def init(world: World, n=1):
    print('init world')
    # set start and end date
    # start = datetime(2023, 1, 1, hour=13)
    # end = datetime(2023, 1, 29, hour=13)
    original_start = datetime(2023, 5, 1, hour=13)
    # original_end   = datetime(2023, 6, 4, hour=13)
    # original_end = datetime(2023, 11, 26, hour=13)  # 47 weeks after original_start
    # original_end = datetime(2023, 1, 8, hour=13)

    SHIFT_WEEKS = 1  # 4 weeks = same weekday
    shift = timedelta(weeks=SHIFT_WEEKS)

    start = original_start + shift
    # end   = original_end + shift
    # end = datetime(2024, 1, 5, hour=13)
    end = datetime(2023, 7, 5, hour=13)

    # create index
    index = FastIndex(start, end, freq="h")

    # set simulation ID
    simulation_id = "mas4te_simulation_test"

    # add possible bidding strategies
    world.bidding_strategies["battery_market_strategy"] = BatteryMarketStrategy
    world.bidding_strategies["community_eom_strategy"] = CommunityEOMStrategy

    # add possible clearing mechanism
    world.clearing_mechanisms["battery_clearing"] = BatteryClearing
    world.clearing_mechanisms["pay_as_clear"] = PayAsClearRole

    # set up world
    world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=simulation_id,
    )

    # create market design
    battery_market = MarketConfig(
        market_id="BatteryMarket",
        opening_hours=rr.rrule(
            rr.WEEKLY,
            interval=1,
            dtstart=start,
            until=end,
            cache=True,  # weekly battery market with the next week tradeable
        ),
        opening_duration=timedelta(hours=1),
        market_mechanism="battery_clearing",
        product_type="power",
        market_products=[
            MarketProduct(
                duration=timedelta(
                    hours=24 * 7
                ),  # each product (storage rent) will be 1 week
                count=1,  # we will only trade the next week, not any week after that
                first_delivery=timedelta(hours=12),
            )
        ],  # delivery will take place 12 hours after market close
        additional_fields=["c_rate", "blockchain_address"],
        param_dict={"allowed_c_rates": [1]},
        minimum_bid_price=0,
        volume_unit="kW",
    )

    # community energy-only market: energy in 15-minute blocks, pay-as-clear
    community_eom = MarketConfig(
        market_id="CommunityEOM",
        opening_hours=rr.rrule(
            rr.HOURLY, interval=1, dtstart=start, until=end, cache=True
        ),
        opening_duration=timedelta(hours=1),
        market_mechanism="pay_as_clear",
        product_type="power",
        market_products=[
            MarketProduct(
                duration=timedelta(
                    minutes=15,
                ),
                count=4 * 24,  # next day in 15-minute-blocks
                first_delivery=timedelta(minutes=15),
            )
        ],
        additional_fields=["blockchain_address"],
        maximum_bid_price=3,
        minimum_bid_price=-0.5,
        volume_unit="kW",
        price_unit="€/kWh",
    )

    # market and the strategy the units bid on it with
    markets = {battery_market.market_id: (battery_market, "battery_market_strategy")}
    if ENABLE_COMMUNITY_EOM:
        markets[community_eom.market_id] = (community_eom, "community_eom_strategy")

    marketdesign = [market for market, _ in markets.values()]
    bidding_strategies = {market_id: strategy for market_id, (_, strategy) in markets.items()}

    # create and add market operator
    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)

    # add market to world
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    ##################################################
    # SET THE NUMBER OF DEMAND AND SUPPLY UNITS HERE #
    ##################################################
    n_demand_units = 2
    n_supply_units = 2
    # the first n_ev_households demand units (households) own an EV, which is
    # modelled as flexible load in the BUC (not sold on the storage market);
    # profile and vehicle type cycle through EV_PROFILES / EV_TYPES_USED
    n_ev_households = 1
    EV_PROFILES = ["0", "1"]  # profile IDs in example_data/ev.csv
    EV_TYPES_USED = ["vw_eup", "tesla"]

    # actually create and add the demand units
    for i in range(n_demand_units):
        print('demand unit ', i)
        # we need a demand, solar generation and price forecast to build bids
        # so we have to read them in before providing them to the forecaster of the unit
        # -----------------------------------------------------------------------------------
        # you can provide an ID (0 to 29) here and the forecast for that ID will be read in
        # or you can set "randomize" to True, to choose a random forecast
        # note: the demand must be the household load WITHOUT EV charging,
        # the EV charging is decided by the BUC (flexible load)
        forecasts = read_forecasts(start, end, id=str(i), randomize=False)

        id_buy = "0" + str(i + 1) if i < 9 else str(i + 1)

        electric_vehicle = None
        ev_forecasts = {}
        if i < n_ev_households:
            ev_forecasts = read_ev_profile(index, id=EV_PROFILES[i % len(EV_PROFILES)])
            electric_vehicle = create_electric_vehicle(
                id=f"EV_{id_buy}",
                ev_type=EV_TYPES_USED[i % len(EV_TYPES_USED)],
                ev_profile=ev_forecasts,
                min_departure_soc=0.6,
            )
        world.add_unit_operator(id=f"storage_demand_operator_{id_buy}")
        world.add_unit(
            id=f"B_{id_buy}",
            # id = id_buy,
            unit_type="mas4te",
            unit_operator_id=f"storage_demand_operator_{id_buy}",
            unit_params={
                "baseline_storage": Storage(
                    id=0, c_rate=1, volume=0, charge_efficiency=1
                ),  # unit has no storage
                "electric_vehicle": electric_vehicle,  # None = household without EV
                "max_power": 1000,  # max 1.000 kW demand
                "min_power": 0,  # no minimum demand
                "bidding_strategies": bidding_strategies,
                "bidding_params": {
                    "role": "buy",
                    # "unit_id" : id_buy,
                    # fallback only, every auction passes its own market config
                    "market_config": battery_market,
                },
                "technology": "demand",
            },
            forecaster=NaiveForecast(
                index=index,
                demand=0,
                energy_demand=forecasts["demand"],
                wholesale_price=forecasts["wholesale_price"],
                eeg_price=forecasts["eeg_price"],
                community_price=forecasts["community_price"],
                grid_price=forecasts["grid_price"],
                solar_gen=forecasts["solar_gen"],
                # ev_availability / ev_driving_consumption_kw, only for EV households
                **ev_forecasts,
            ),
        )

    # actually create and add the supply units
    for i in range(n_supply_units):
        print('supply unit ', i)
        # same as above - set an ID or set randomize to True
        forecasts = read_forecasts(start, end, id=str(i))

        id_sell = "0" + str(i + 1) if i < 9 else str(i + 1)
        # id_sell = id_sell = str(i).zfill(3)
        world.add_unit_operator(f"storage_provider_operator_{id_sell}")
        world.add_unit(
            # id=f"storage_provider_{id_sell}",
            id=f"S_{id_sell}",
            unit_type="mas4te",
            unit_operator_id=f"storage_provider_operator_{id_sell}",
            unit_params={
                "baseline_storage": Storage(
                    id=0, c_rate=1, volume=5, charge_efficiency=0.95
                ),
                "max_power_charge": 1,  # max 1 kW charge
                "max_power_discharge": 1,  # max 1 kW discharge
                "max_soc": 20,  # max 20 kWh of storage capacity (equal to baseline)
                "min_soc": 0,  # no mimimum fill level
                "efficiency_charge": 0.975,  # charge and discharge to combine to 95% efficiency
                "efficiency_discharge": 0.975,
                "bidding_strategies": bidding_strategies,
                "bidding_params": {
                    "role": "sell",
                    # "unit_id" : id_sell,
                    # fallback only, every auction passes its own market config
                    "market_config": battery_market,
                },
                "technology": "battery_storage",
            },
            forecaster=NaiveForecast(
                index=index,
                demand=0,
                availability=1,  # always available
                energy_demand=forecasts["demand"],
                solar_gen=forecasts["solar_gen"],
                wholesale_price=forecasts["wholesale_price"],
                eeg_price=forecasts["eeg_price"],
                community_price=forecasts["community_price"],
                grid_price=forecasts["grid_price"],
                # no battery demand, fuel price or CO2 price for this simulation
                battery_demand=0,  # no battery demand
                fuel_price=0,  # no fuel price
                co2_price=0,  # no CO2 price
            ),
        )


if __name__ == "__main__":
    # BRP_MARKET_RESULTS_URL and MARKET_RESULTS_PASSWORD, see example.env
    load_dotenv()
    # db_uri = "postgresql://assume:assume@localhost:5432/assume"
    db_uri = "sqlite:///new-mas4te-assume.db"
    world = World(database_uri=db_uri, log_level="ERROR")
    init(world)
    logging.getLogger("gurobipy").setLevel(logging.WARNING)  # suppress gurobipy logs
    world.run()
