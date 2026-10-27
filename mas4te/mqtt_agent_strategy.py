# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import time
from datetime import datetime

import paho.mqtt.client as mqtt

from assume.common.base import BaseStrategy, BaseUnit
from assume.common.market_objects import MarketConfig, Orderbook
from assume.units.mas4te import electric_vehicle_params


class MQTTAgentStrategy(BaseStrategy):
    """
    MQTT-based strategy that can act as buyer or seller.
    Supports multi-bid orderbooks.

    The bids come from an external AI agent: the strategy publishes
    ``market_open``, waits for the bids and sends back the market result.
    Subclasses set the MQTT topics of their market (``TOPIC_PREFIX``) and turn
    the bids into orders (``make_orders``).
    """

    # topics: <TOPIC_PREFIX>/market/status_agent<id>, <TOPIC_PREFIX>/bids/agent<id>,
    # <TOPIC_PREFIX>/results/agent<id>
    TOPIC_PREFIX = "mas4te"

    # Shared MQTT clients across all instances
    _mqtt_clients = {}

    def __init__(self, unit_id, role="buy", market_config=None, comm_agent_port=8000):
        self.unit_id = unit_id
        self.role = role  # "buy" or "sell"
        self.market_config = market_config
        self.latest_bids = None  # <-- now stores multiple bids
        self.market_open_sent = False

        # MQTT topics
        self.TOPIC_MARKET_STATUS = f"{self.TOPIC_PREFIX}/market/status_agent{str(self.unit_id)}"
        self.TOPIC_BIDS = f"{self.TOPIC_PREFIX}/bids/agent{str(self.unit_id)}"
        self.TOPIC_RESULTS = f"{self.TOPIC_PREFIX}/results/agent{str(self.unit_id)}"

        # MQTT setup (shared clients, one per market strategy of a unit)
        client_key = f"{self.TOPIC_PREFIX}_{role}_{unit_id}_{comm_agent_port}"
        if client_key not in MQTTAgentStrategy._mqtt_clients:
            self.client = mqtt.Client(
                client_id=f"assume_{self.TOPIC_PREFIX.replace('/', '_')}_{comm_agent_port}_{role}_{unit_id}"
            )
            self.client.on_message = self.on_message
            self.client.on_connect = self.on_connect
            self.client.connect("localhost", 1883, 60)
            self.client.loop_start()
            time.sleep(0.5)  # wait for connection to establish
            print(f"{str(self.unit_id)}, Subscribed to {self.TOPIC_BIDS}")
            self.run_loop = True
            MQTTAgentStrategy._mqtt_clients[client_key] = self.client
        else:
            self.client = MQTTAgentStrategy._mqtt_clients[client_key]

    def build_market_open_msg(
        self, unit, product_tuples: list[tuple], market_config=None
    ) -> dict:
        """
        Build the ``market_open`` message for the AI agent of ``unit``.

        Besides the market configuration and the open products it contains the
        optional field ``electric_vehicle`` with the static EV parameters of the
        unit (see ``assume.units.mas4te.electric_vehicle_params``) or None if
        the unit has no EV. Agents that do not know the field can ignore it.
        """
        market_config = market_config or self.market_config

        products_payload = []
        for p in product_tuples:
            start_time, end_time, only_hours = p
            products_payload.append(
                {
                    "start_time": start_time.isoformat()
                    if hasattr(start_time, "isoformat")
                    else str(start_time),
                    "end_time": end_time.isoformat()
                    if hasattr(end_time, "isoformat")
                    else str(end_time),
                    "only_hours": only_hours,
                }
            )

        # Build market products from config
        market_products_payload = []
        for p in getattr(market_config, "market_products", []):
            market_products_payload.append(
                {
                    "duration_seconds": p.duration.total_seconds(),
                    "count": p.count,
                    "first_delivery": p.first_delivery.total_seconds(),
                    "only_hours": p.only_hours,
                }
            )

        return {
            "status": "market_open",
            "market_id": market_config.market_id,
            "product_type": market_config.product_type,
            "maximum_bid_volume": market_config.maximum_bid_volume,
            "maximum_bid_price": market_config.maximum_bid_price,
            "minimum_bid_price": market_config.minimum_bid_price,
            "volume_unit": market_config.volume_unit,
            "price_unit": market_config.price_unit,
            "additional_fields": market_config.additional_fields,
            "market_products": market_products_payload,
            "products": products_payload,
            "electric_vehicle": electric_vehicle_params(
                getattr(unit, "electric_vehicle", None)
            ),
        }

    def calculate_bids(
        self, unit, product_tuples: list[tuple], market_config=None, **kwargs
    ):
        """
        Wait for bids from agent and return them as an Orderbook (list of orders).
        """
        # every auction passes its own market config; the one from the
        # bidding params is only the fallback
        market_config = market_config or self.market_config

        self.run_loop = True
        self.latest_bids = None

        print(f"[DEBUG] {str(self.unit_id)}  calculate_bids called for {self.unit_id}")
        print(f"[DEBUG] {str(self.unit_id)}  MQTT connected: {self.client.is_connected()}")

        # Send market open message on first call (when we have products)
        if not self.market_open_sent and market_config:
            market_open_msg = self.build_market_open_msg(
                unit, product_tuples, market_config
            )

            msg_info = self.client.publish(
                self.TOPIC_MARKET_STATUS,
                json.dumps(market_open_msg, indent=4),
                qos=1,
                retain=True,
            )
            msg_info.wait_for_publish()
            self.market_open_sent = True
            print(f" {str(self.unit_id)}  {datetime.now().isoformat()}, Market open message sent by {self.role} strategy")
            time.sleep(0.2)  # Give MQTT time to propagate

        max_run_time = 10 * 60
        start_time = time.time()
        while self.run_loop:
            if (time.time() - start_time) > max_run_time:
                print(f"[DEBUG] {str(self.unit_id)} Timeout waiting for bids for {self.unit_id}")
                break
            time.sleep(0.1)

        if not self.latest_bids:
            print(f"{str(self.unit_id)} No bids received for {self.unit_id}, returning empty orderbook")
            return []

        return self.make_orders(unit, self.latest_bids, product_tuples, market_config)

    def make_orders(self, unit, bids: list, product_tuples: list[tuple], market_config) -> list:
        """Turn the bids of the agent into orders of the market."""
        raise NotImplementedError

    def base_order(self, unit, bid: dict, product: tuple, volume: float) -> dict:
        """Fields every order has, whatever the market."""
        start_time, end_time, only_hours = product
        return {
            # prefix with the unit id, agents number their bids independently
            # and the clearing needs unique bid ids to report the matches
            "bid_id": f"{unit.id}_{bid.get('bid_id', 0)}",
            "start_time": start_time,
            "end_time": end_time,
            "volume": volume,
            "price": bid.get("price", 0),
            "agent_addr": getattr(unit, "addr", "agent_addr_unknown"),
            "node": getattr(unit, "node", "node_unknown"),
            "only_hours": only_hours if only_hours is not None else [],
        }

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"{str(self.unit_id)} {self.role.capitalize()} strategy connected to MQTT broker, rc={rc}")
        client.subscribe([(self.TOPIC_BIDS, 0)])

    def on_message(self, client, userdata, msg):
        if msg.topic == self.TOPIC_BIDS:
            payload = json.loads(msg.payload.decode())

            # Normalize to list
            if isinstance(payload, dict):
                bids = [payload]
            elif isinstance(payload, list):
                bids = payload
            else:
                print(f"Invalid payload type: {type(payload)}")
                return

            self.latest_bids = bids

            # Acknowledge
            client.publish(
                self.TOPIC_RESULTS,
                json.dumps(
                    {
                        "ack": "bids_received",
                        "count": len(bids),
                        "bid_ids": [
                            b.get("bid_id", 0) for b in bids if isinstance(b, dict)
                        ],
                    }
                ),
            )

            print("Confirmed bids received")

            self.run_loop = False

    def calculate_reward(
        self, unit: BaseUnit, marketconfig: MarketConfig, orderbook: Orderbook
    ):
        print("in calculate reward")
        self.market_open_sent = False

        # the clearing stores the counterparties as JSON string, decode them
        # so the agent gets who it was matched with as structured data
        orders = []
        for order in orderbook:
            order = dict(order)
            if isinstance(order.get("matches"), str):
                order["matches"] = json.loads(order["matches"])
            orders.append(order)

        payload = {
            "msg": "market result",
            "market_id": getattr(marketconfig, "market_id", None),
            "orderbook": orders,
            "unit_id": getattr(unit, "id", None),
        }

        # Publish using the existing MQTT client and topic
        if hasattr(self, "client") and hasattr(self, "TOPIC_RESULTS"):
            msg_info = self.client.publish(
                self.TOPIC_RESULTS,
                json.dumps(payload, default=str, indent=4),
                qos=1,
                retain=False,
            )
            msg_info.wait_for_publish()
            print(
                f"Market reward sent via MQTT on {self.TOPIC_RESULTS} for unit {unit.id}"
            )
        else:
            print("MQTT client or results topic not initialized, cannot send reward.")
