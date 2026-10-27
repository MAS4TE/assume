# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the strategies of the battery market and the community energy market."""

import importlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct

# the mas4te scripts import their siblings directly, like simulation.py
MAS4TE_DIR = Path(__file__).parent.parent / "mas4te"
sys.path.insert(0, str(MAS4TE_DIR))
mqtt_agent_strategy = importlib.import_module("mqtt_agent_strategy")
BatteryMarketStrategy = importlib.import_module("battery_market_strategy").BatteryMarketStrategy
CommunityEOMStrategy = importlib.import_module("community_eom_strategy").CommunityEOMStrategy

START = datetime(2023, 5, 8, 13)
END = datetime(2023, 5, 22, 13)

battery_market = MarketConfig(
    market_id="BatteryMarket",
    opening_hours=rr.rrule(rr.WEEKLY, dtstart=START, until=END),
    opening_duration=timedelta(hours=1),
    market_mechanism="battery_clearing",
    product_type="power",
    market_products=[
        MarketProduct(
            duration=timedelta(hours=24 * 7), count=1, first_delivery=timedelta(hours=12)
        )
    ],
    additional_fields=["c_rate", "blockchain_address"],
    param_dict={"allowed_c_rates": [1]},
    minimum_bid_price=0,
    volume_unit="kW",
)

community_eom = MarketConfig(
    market_id="CommunityEOM",
    opening_hours=rr.rrule(rr.HOURLY, dtstart=START, until=END),
    opening_duration=timedelta(hours=1),
    market_mechanism="pay_as_clear",
    product_type="power",
    market_products=[
        MarketProduct(
            duration=timedelta(minutes=15), count=4, first_delivery=timedelta(minutes=15)
        )
    ],
    additional_fields=["blockchain_address"],
    maximum_bid_price=3,
    minimum_bid_price=-0.5,
    volume_unit="kW",
    price_unit="€/kWh",
)

BATTERY_PRODUCTS = [(START + timedelta(hours=12), START + timedelta(hours=12 + 24 * 7), None)]
EOM_PRODUCTS = [
    (START + timedelta(minutes=15 * k), START + timedelta(minutes=15 * (k + 1)), None)
    for k in range(1, 5)
]


@pytest.fixture
def mqtt_clients(monkeypatch):
    """Every strategy gets its own mocked MQTT client."""
    created = []

    def new_client(*args, **kwargs):
        client = MagicMock()
        created.append(client)
        return client

    monkeypatch.setattr(mqtt_agent_strategy.mqtt, "Client", new_client)
    monkeypatch.setattr(mqtt_agent_strategy.time, "sleep", lambda *_: None)
    monkeypatch.setattr(mqtt_agent_strategy.MQTTAgentStrategy, "_mqtt_clients", {})
    return created


def _unit():
    return SimpleNamespace(id="P_01", addr="addr", node="node0", electric_vehicle=None)


def _agent_answers(strategy, answer):
    """Let the mocked agent answer the market_open of ``strategy`` with ``answer(market_open)``."""
    published = []

    def publish(topic, payload, *args, **kwargs):
        published.append((topic, payload))
        if topic == strategy.TOPIC_MARKET_STATUS:
            reply = answer(json.loads(payload))
            strategy.on_message(
                strategy.client,
                None,
                SimpleNamespace(topic=strategy.TOPIC_BIDS, payload=json.dumps(reply).encode()),
            )
        return MagicMock()

    strategy.client.publish.side_effect = publish
    return published


def test_markets_have_their_own_topics_and_clients(mqtt_clients):
    battery = BatteryMarketStrategy(unit_id="P_01", role="buy", market_config=battery_market)
    eom = CommunityEOMStrategy(unit_id="P_01", role="buy", market_config=battery_market)

    # the battery market keeps the topics the agents already use
    assert battery.TOPIC_MARKET_STATUS == "mas4te/market/status_agentP_01"
    assert battery.TOPIC_BIDS == "mas4te/bids/agentP_01"
    assert eom.TOPIC_MARKET_STATUS == "mas4te/community_eom/market/status_agentP_01"
    assert eom.TOPIC_BIDS == "mas4te/community_eom/bids/agentP_01"
    assert eom.TOPIC_RESULTS == "mas4te/community_eom/results/agentP_01"
    assert len(mqtt_clients) == 2 and battery.client is not eom.client


def test_battery_market_bids(mqtt_clients):
    strategy = BatteryMarketStrategy(unit_id="P_01", role="buy", market_config=battery_market)
    _agent_answers(
        strategy,
        lambda msg: [
            {"bid_id": 1, "quantity": 4, "price": 0.1, "blockchain_address": "0xab"},
            {"bid_id": 2, "volume": 2, "price": 0.05, "blockchain_address": "0xab"},
            {"bid_id": 3, "quantity": 1, "price": 0.2},  # no address, skipped
        ],
    )
    orders = strategy.calculate_bids(_unit(), BATTERY_PRODUCTS, market_config=battery_market)
    assert [(o["bid_id"], o["volume"], o["c_rate"]) for o in orders] == [
        ("P_01_1", -4, 1),
        ("P_01_2", -2, 1),
    ]


def test_eom_market_open_uses_the_eom_config(mqtt_clients):
    # the bidding params of a unit carry the battery market; the auction passes its own config
    strategy = CommunityEOMStrategy(unit_id="P_01", role="buy", market_config=battery_market)
    published = _agent_answers(strategy, lambda msg: [])
    strategy.calculate_bids(_unit(), EOM_PRODUCTS, market_config=community_eom)

    market_open = [json.loads(p) for t, p in published if t == strategy.TOPIC_MARKET_STATUS][0]
    assert market_open["market_id"] == "CommunityEOM"
    assert market_open["price_unit"] == "€/kWh"
    assert len(market_open["products"]) == 4
    assert market_open["market_products"][0]["duration_seconds"] == 15 * 60


def test_eom_bids_per_period_and_direction(mqtt_clients):
    strategy = CommunityEOMStrategy(unit_id="P_01", role="buy", market_config=battery_market)

    def answer(msg):
        first, second, third = (p["start_time"] for p in msg["products"][:3])
        return [
            # PV surplus sold, deficit bought, in different quarter hours
            {"bid_id": 1, "start_time": first, "side": "sell", "quantity": 1.5,
             "price": 0.08, "blockchain_address": "0xab"},
            {"bid_id": 2, "start_time": second, "volume": -0.7, "price": 0.3,
             "blockchain_address": "0xab"},
            # no side and no volume: the role (buy) sets the sign
            {"bid_id": 3, "start_time": third, "quantity": 0.2, "price": 0.3,
             "blockchain_address": "0xab"},
            # skipped: period not open, no address
            {"bid_id": 4, "start_time": "2030-01-01T00:00:00", "quantity": 1,
             "price": 0.1, "blockchain_address": "0xab"},
            {"bid_id": 5, "start_time": first, "quantity": 1, "price": 0.1},
        ]

    _agent_answers(strategy, answer)
    orders = strategy.calculate_bids(_unit(), EOM_PRODUCTS, market_config=community_eom)

    assert [(o["bid_id"], o["start_time"], o["volume"]) for o in orders] == [
        ("P_01_1", EOM_PRODUCTS[0][0], 1.5),
        ("P_01_2", EOM_PRODUCTS[1][0], -0.7),
        ("P_01_3", EOM_PRODUCTS[2][0], -0.2),
    ]
    assert all(o["end_time"] - o["start_time"] == timedelta(minutes=15) for o in orders)
    assert all("c_rate" not in o and o["blockchain_address"] == "0xab" for o in orders)


def test_market_result_on_the_market_topic(mqtt_clients):
    strategy = CommunityEOMStrategy(unit_id="P_01", role="buy", market_config=battery_market)
    strategy.market_open_sent = True
    strategy.calculate_reward(_unit(), community_eom, [])
    topic, payload = strategy.client.publish.call_args[0][:2]
    assert topic == "mas4te/community_eom/results/agentP_01"
    assert json.loads(payload)["market_id"] == "CommunityEOM"
    assert strategy.market_open_sent is False


def test_community_eom_is_off_by_default():
    simulation = importlib.import_module("simulation")
    assert simulation.ENABLE_COMMUNITY_EOM is False
