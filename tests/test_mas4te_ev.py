# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the electric vehicle (EV) support of the MAS4TE units and LLMStrategy."""

import importlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from battery_utility_calculator import Storage
from dateutil import rrule as rr

from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.units.mas4te import MAS4TEUnit, electric_vehicle_params
from mas4te import mas4te_bidding_strategy as strategy_module
from mas4te.mas4te_bidding_strategy import LLMStrategy

try:
    from battery_utility_calculator import ElectricVehicle
except ImportError:  # BUC without EV support
    ElectricVehicle = None

requires_buc_ev = pytest.mark.skipif(
    ElectricVehicle is None,
    reason="installed battery_utility_calculator has no ElectricVehicle",
)

MAS4TE_DIR = Path(__file__).parent.parent / "mas4te"
START = datetime(2023, 5, 8, 13)
END = datetime(2023, 5, 22, 13)

# keys of the market_open message before the EV extension
LEGACY_MARKET_OPEN_KEYS = {
    "status",
    "market_id",
    "product_type",
    "maximum_bid_volume",
    "maximum_bid_price",
    "minimum_bid_price",
    "volume_unit",
    "price_unit",
    "additional_fields",
    "market_products",
    "products",
}

market_config = MarketConfig(
    market_id="BatteryMarket",
    opening_hours=rr.rrule(rr.WEEKLY, dtstart=START, until=END),
    opening_duration=timedelta(hours=1),
    market_mechanism="battery_clearing",
    product_type="power",
    market_products=[
        MarketProduct(
            duration=timedelta(hours=24 * 7),
            count=1,
            first_delivery=timedelta(hours=12),
        )
    ],
    additional_fields=["c_rate", "blockchain_address"],
    param_dict={"allowed_c_rates": [1]},
    minimum_bid_price=0,
    volume_unit="kW",
)
PRODUCTS = [(START + timedelta(hours=12), START + timedelta(hours=12 + 24 * 7), None)]


def _index():
    return FastIndex(START, END, freq="h")


def _ev_series(index):
    idx = index.as_datetimeindex()
    # away 07-17 on weekdays, 4 kW while driving in the first and last hour
    away = (idx.day_of_week < 5) & (idx.hour >= 7) & (idx.hour < 17)
    availability = pd.Series((~away).astype(int), index=idx)
    driving = pd.Series(0.0, index=idx)
    driving[away & ((idx.hour == 7) | (idx.hour == 16))] = 4.0
    return availability, driving


def _stub_ev(**overrides):
    """Duck-typed stand-in with the attributes of the BUC ElectricVehicle."""
    params = dict(
        id="EV_01",
        capacity_kwh=32.3,
        max_charge_kw=7.2,
        max_discharge_kw=0.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        initial_soc=0.5,
        min_soc=0.0,
        min_departure_soc=0.6,
        public_charging_price=None,
        availability=pd.Series([1, 0]),
        driving_consumption=pd.Series([0.0, 4.0]),
    )
    params.update(overrides)
    return SimpleNamespace(**params)


def _unit(electric_vehicle=None, **kwargs):
    return MAS4TEUnit(
        id="B_01",
        unit_operator="storage_demand_operator_01",
        technology="demand",
        bidding_strategies={},
        forecaster=NaiveForecast(index=_index(), demand=0),
        baseline_storage=Storage(id=0, c_rate=1, volume=0, charge_efficiency=1),
        electric_vehicle=electric_vehicle,
        **kwargs,
    )


@pytest.fixture
def mqtt_client(monkeypatch):
    """Replace the paho MQTT client, there is no broker in the tests."""
    client = MagicMock()
    client.is_connected.return_value = True
    monkeypatch.setattr(strategy_module.mqtt, "Client", MagicMock(return_value=client))
    monkeypatch.setattr(strategy_module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(LLMStrategy, "_mqtt_clients", {})
    return client


def _strategy(role="buy"):
    return LLMStrategy(unit_id="B_01", role=role, market_config=market_config)


# ---------------------------------------------------------------- MAS4TEUnit


def test_unit_without_ev_is_backwards_compatible():
    # old call without the electric_vehicle argument still works
    unit = MAS4TEUnit(
        id="S_01",
        unit_operator="op",
        technology="battery_storage",
        bidding_strategies={},
        forecaster=NaiveForecast(index=_index(), demand=0),
        baseline_storage=Storage(id=0, c_rate=1, volume=5),
    )
    assert unit.electric_vehicle is None
    unit_dict = unit.as_dict()
    assert unit_dict["electric_vehicle"] is None
    assert unit_dict["baseline_storage"] == str(unit.baseline_storage)


def test_unit_with_ev_stub_as_dict():
    ev = _stub_ev()
    unit = _unit(electric_vehicle=ev)
    assert unit.electric_vehicle is ev

    ev_dict = json.loads(unit.as_dict()["electric_vehicle"])
    assert ev_dict == {
        "id": "EV_01",
        "capacity_kwh": 32.3,
        "max_charge_kw": 7.2,
        "max_discharge_kw": 0.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "initial_soc": 0.5,
        "min_soc": 0.0,
        "min_departure_soc": 0.6,
        "public_charging_price": None,
        "v2h": False,
    }
    # no time series in the static parameters
    assert "availability" not in ev_dict
    assert "driving_consumption" not in ev_dict


def test_electric_vehicle_params():
    assert electric_vehicle_params(None) is None
    params = electric_vehicle_params(
        _stub_ev(id=("hh", 3), max_discharge_kw=3.7, public_charging_price=0.59)
    )
    assert params["id"] == "('hh', 3)"
    assert params["v2h"] is True
    assert params["max_discharge_kw"] == 3.7
    assert params["public_charging_price"] == 0.59
    json.dumps(params)  # serialisable


@requires_buc_ev
def test_unit_with_buc_electric_vehicle():
    availability, driving = _ev_series(_index())
    ev = ElectricVehicle(
        id="EV_01",
        capacity_kwh=57.5,
        max_charge_kw=11,
        availability=availability,
        driving_consumption=driving,
        max_discharge_kw=3.7,
        min_departure_soc=0.6,
    )
    unit = _unit(electric_vehicle=ev)
    ev_dict = json.loads(unit.as_dict()["electric_vehicle"])
    assert ev_dict["capacity_kwh"] == 57.5
    assert ev_dict["max_charge_kw"] == 11.0
    assert ev_dict["max_discharge_kw"] == 3.7
    assert ev_dict["min_departure_soc"] == 0.6
    assert ev_dict["v2h"] is True
    assert ev_dict["initial_soc"] == ev.initial_soc


# ------------------------------------------------------- market_open payload


def test_market_open_without_ev(mqtt_client):
    msg = _strategy().build_market_open_msg(_unit(), PRODUCTS)
    # all previous fields are unchanged, electric_vehicle is the only addition
    assert set(msg) == LEGACY_MARKET_OPEN_KEYS | {"electric_vehicle"}
    assert msg["electric_vehicle"] is None
    assert msg["status"] == "market_open"
    assert msg["market_id"] == "BatteryMarket"
    assert msg["products"] == [
        {
            "start_time": PRODUCTS[0][0].isoformat(),
            "end_time": PRODUCTS[0][1].isoformat(),
            "only_hours": None,
        }
    ]
    assert msg["market_products"][0]["duration_seconds"] == 7 * 24 * 3600
    assert json.loads(json.dumps(msg))["electric_vehicle"] is None


def test_market_open_with_ev(mqtt_client):
    msg = _strategy().build_market_open_msg(
        _unit(electric_vehicle=_stub_ev()), PRODUCTS
    )
    assert LEGACY_MARKET_OPEN_KEYS <= set(msg)
    ev = json.loads(json.dumps(msg))["electric_vehicle"]
    assert ev["capacity_kwh"] == 32.3
    assert ev["max_charge_kw"] == 7.2
    assert ev["min_departure_soc"] == 0.6


def test_market_open_unit_without_attribute(mqtt_client):
    # any unit type without the attribute (e.g. other BaseUnits) gives null
    unit = SimpleNamespace(id="X")
    assert _strategy().build_market_open_msg(unit, PRODUCTS)["electric_vehicle"] is None


@pytest.mark.parametrize("with_ev", [False, True])
def test_calculate_bids_publishes_market_open(mqtt_client, with_ev):
    strategy = _strategy()
    unit = _unit(electric_vehicle=_stub_ev() if with_ev else None)
    published = []

    def publish(topic, payload, *args, **kwargs):
        published.append((topic, payload))
        if topic == strategy.TOPIC_MARKET_STATUS:
            # the AI agent answers the market_open with its bids
            bid = {"bid_id": 1, "quantity": 2, "price": 3, "blockchain_address": "0xab"}
            strategy.on_message(
                mqtt_client,
                None,
                SimpleNamespace(
                    topic=strategy.TOPIC_BIDS, payload=json.dumps([bid]).encode()
                ),
            )
        return MagicMock()

    mqtt_client.publish.side_effect = publish

    orders = strategy.calculate_bids(unit, PRODUCTS, market_config=market_config)

    market_open = [
        json.loads(p) for t, p in published if t == strategy.TOPIC_MARKET_STATUS
    ]
    assert len(market_open) == 1
    assert market_open[0]["status"] == "market_open"
    if with_ev:
        assert market_open[0]["electric_vehicle"]["capacity_kwh"] == 32.3
    else:
        assert market_open[0]["electric_vehicle"] is None

    assert len(orders) == 1
    assert orders[0]["bid_id"] == "B_01_1"
    assert orders[0]["volume"] == -2


# ------------------------------------------------------- example scenario data


@pytest.fixture
def simulation(monkeypatch):
    # simulation.py is a script that imports its siblings directly
    monkeypatch.syspath_prepend(str(MAS4TE_DIR))
    return importlib.import_module("simulation")


def test_read_ev_profile_placeholder(simulation):
    index = _index()
    profiles = {
        i: simulation.read_ev_profile(
            index, id=i, data_dir=str(MAS4TE_DIR / "example_data")
        )
        for i in ("0", "1")
    }
    for profile in profiles.values():
        availability = profile["ev_availability"]
        driving = profile["ev_driving_consumption_kw"]
        assert len(availability) == len(driving) == len(index)
        assert (availability.index == index.as_datetimeindex()).all()
        assert set(availability.unique()) <= {0.0, 1.0}
        assert (driving >= 0).all()
        # no driving while at home, and the car is away at some point
        assert (driving[availability == 1] == 0).all()
        assert (availability == 0).any() and (driving > 0).any()

    # commuter profile 0: Monday 2023-05-08 away from 07 to 17 (placeholder week repeated)
    a0 = profiles["0"]["ev_availability"]
    assert a0[datetime(2023, 5, 15, 9)] == 0
    assert a0[datetime(2023, 5, 15, 20)] == 1


def test_read_ev_profile_single_vehicle_file(simulation, tmp_path):
    # file per vehicle with the unsuffixed contract columns, covering the index
    idx = _index().as_datetimeindex()
    availability, driving = _ev_series(_index())
    pd.DataFrame(
        {"ev_availability": availability, "ev_driving_consumption_kw": driving},
        index=idx,
    ).to_csv(tmp_path / "ev.csv")

    profile = simulation.read_ev_profile(_index(), id="7", data_dir=str(tmp_path))
    pd.testing.assert_series_equal(
        profile["ev_availability"], availability.astype(float), check_names=False
    )
    pd.testing.assert_series_equal(
        profile["ev_driving_consumption_kw"], driving, check_names=False
    )


@requires_buc_ev
def test_create_electric_vehicle(simulation):
    profile = simulation.read_ev_profile(
        _index(), id="0", data_dir=str(MAS4TE_DIR / "example_data")
    )
    ev = simulation.create_electric_vehicle(
        "EV_01", "vw_eup", profile, min_departure_soc=0.6
    )
    assert isinstance(ev, ElectricVehicle)
    assert ev.capacity_kwh == simulation.EV_TYPES["vw_eup"]["capacity_kwh"]
    assert ev.min_departure_soc == 0.6
    assert len(ev.availability) == len(_index())
