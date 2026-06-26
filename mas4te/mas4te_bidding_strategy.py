# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import time

import paho.mqtt.client as mqtt

from assume.common.base import BaseStrategy, BaseUnit
from assume.common.market_objects import MarketConfig, Orderbook

from datetime import datetime


class LLMStrategy(BaseStrategy):
    """
    MQTT-based strategy that can act as buyer or seller.
    Supports multi-bid orderbooks.
    """

    # Shared MQTT clients across all instances
    _mqtt_clients = {}

    def __init__(self, unit_id, role="buy", market_config=None, comm_agent_port=8000):
        self.unit_id = unit_id
        self.role = role  # "buy" or "sell"
        self.market_config = market_config
        self.latest_bids = None  # <-- now stores multiple bids
        self.market_open_sent = False

        # MQTT topics
        self.TOPIC_MARKET_STATUS = f"mas4te/market/status_agent{str(self.unit_id)}"
        self.TOPIC_BIDS = f"mas4te/bids/agent{str(self.unit_id)}"
        self.TOPIC_RESULTS = f"mas4te/results/agent{str(self.unit_id)}"

        # MQTT setup (shared clients)
        client_key = f"{role}_{unit_id}_{comm_agent_port}"
        if client_key not in LLMStrategy._mqtt_clients:
            self.client = mqtt.Client(
                client_id=f"assume_{comm_agent_port}_{role}_{unit_id}"
            )
            self.client.on_message = self.on_message
            self.client.on_connect = self.on_connect
            self.client.connect("localhost", 1883, 60)
            self.client.loop_start()
            time.sleep(0.5)  # wait for connection to establish
            # self.client.subscribe([(self.TOPIC_BIDS, 0)])
            print(f"{str(self.unit_id)}, Subscribed to {self.TOPIC_BIDS}")
            self.run_loop = True
            LLMStrategy._mqtt_clients[client_key] = self.client
        else:
            self.client = LLMStrategy._mqtt_clients[client_key]

    def calculate_bids(
        self, unit, product_tuples: list[tuple], market_config=None, **kwargs
    ):
        """
        Wait for bids from agent and return them as an Orderbook (list of orders).
        """

        self.run_loop = True
        self.latest_bids = None
        # self.market_open_sent = False

        print(f"[DEBUG] {str(self.unit_id)}  calculate_bids called for {self.unit_id}")
        print(f"[DEBUG] {str(self.unit_id)}  MQTT connected: {self.client.is_connected()}")

        # self.client.loop_start()
        
        # Send market open message on first call (when we have products)
        if not self.market_open_sent and self.market_config:
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
            for p in getattr(self.market_config, "market_products", []):
                market_products_payload.append(
                    {
                        "duration_seconds": p.duration.total_seconds(),
                        "count": p.count,
                        "first_delivery": p.first_delivery.total_seconds(),
                        "only_hours": p.only_hours,
                    }
                )

            market_open_msg = {
                "status": "market_open",
                "market_id": self.market_config.market_id,
                "product_type": self.market_config.product_type,
                "maximum_bid_volume": self.market_config.maximum_bid_volume,
                "maximum_bid_price": self.market_config.maximum_bid_price,
                "minimum_bid_price": self.market_config.minimum_bid_price,
                "volume_unit": self.market_config.volume_unit,
                "price_unit": self.market_config.price_unit,
                "additional_fields": self.market_config.additional_fields,
                "market_products": market_products_payload,
                "products": products_payload,
            }

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

        # self.client.loop_stop()

        if not self.latest_bids:
            print(f"{str(self.unit_id)} No bids received for {self.unit_id}, returning empty orderbook")
            return []

        orders = []

        start_time, end_time, only_hours = product_tuples[0]
        c_rate = market_config.param_dict["allowed_c_rates"][0]

        for bid in self.latest_bids:
            if not isinstance(bid, dict):
                print(f"Skipping invalid bid: {bid}")
                continue

            quantity = bid.get("quantity", 0)

            # fallback if still using "volume"
            if "volume" in bid and "quantity" not in bid:
                quantity = bid["volume"]

            volume = quantity if self.role == "sell" else -quantity

            order = {
                "bid_id": bid.get("bid_id", 0),
                "start_time": start_time,
                "end_time": end_time,
                "volume": volume,
                "price": bid.get("price", 0),
                "agent_addr": getattr(unit, "addr", "agent_addr_unknown"),
                "node": getattr(unit, "node", "node_unknown"),
                "only_hours": only_hours if only_hours is not None else [],
                "c_rate": c_rate,
            }

            orders.append(order)

        return orders

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"{str(self.unit_id)} {self.role.capitalize()} strategy connected to MQTT broker, rc={rc}")
        client.subscribe([(self.TOPIC_BIDS, 0)])

    def on_message(self, client, userdata, msg):
        # print(f"{str(self.unit_id)} on message" , msg.payload.decode())
        if msg.topic == self.TOPIC_BIDS:
            payload = json.loads(msg.payload.decode())
            # print(f" {str(self.unit_id)} {self.role.capitalize()} strategy received bid(s): {payload}")

            # Normalize to list
            if isinstance(payload, dict):
                bids = [payload]
            elif isinstance(payload, list):
                bids = payload
            else:
                print(f"Invalid payload type: {type(payload)}")
                return

            # Normalize fields
            for bid in bids:
                if isinstance(bid, dict):
                    if "volume" in bid and "quantity" not in bid:
                        bid["quantity"] = bid["volume"]

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

        payload = {
            "msg": "market result",
            "market_id": getattr(marketconfig, "market_id", None),
            "orderbook": orderbook,
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
