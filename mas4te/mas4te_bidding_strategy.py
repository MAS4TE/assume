import time
import json
import paho.mqtt.client as mqtt
from datetime import datetime
from typing import List

from assume.common.market_objects import Orderbook, Product, MarketConfig
from assume.common.base import BaseStrategy, BaseUnit, SupportsMinMaxCharge


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
            self.client = mqtt.Client(client_id=f"assume_{comm_agent_port}_{role}_{unit_id}")
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.connect("localhost", 1883, 60)
            self.client.loop_start()
            LLMStrategy._mqtt_clients[client_key] = self.client
        else:
            self.client = LLMStrategy._mqtt_clients[client_key]

    def calculate_bids(self, unit, product_tuples: List[tuple], market_config=None, **kwargs):
        """
        Wait for bids from agent and return them as an Orderbook (list of orders).
        """

        # ---- MARKET OPEN MESSAGE ----
        if not self.market_open_sent and self.market_config:
            products_payload = []
            for p in product_tuples:
                start_time, end_time, only_hours = p
                products_payload.append({
                    "start_time": start_time.isoformat() if hasattr(start_time, 'isoformat') else str(start_time),
                    "end_time": end_time.isoformat() if hasattr(end_time, 'isoformat') else str(end_time),
                    "only_hours": only_hours,
                })

            market_products_payload = []
            for p in getattr(self.market_config, "market_products", []):
                market_products_payload.append({
                    "duration_seconds": p.duration.total_seconds(),
                    "count": p.count,
                    "first_delivery": p.first_delivery.total_seconds(),
                    "only_hours": p.only_hours,
                })

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
                retain=True
            )
            msg_info.wait_for_publish()
            self.market_open_sent = True
            print(f"Market open message sent by {self.role} strategy")
            time.sleep(0.2)

        # ---- WAIT FOR BIDS ----
        print(f"{self.role.capitalize()} strategy waiting for bids from agent...")
        timeout = 4000
        start = time.time()

        while self.latest_bids is None and (time.time() - start < timeout):
            time.sleep(0.05)

        if self.latest_bids is None:
            print(f"No bids received within timeout for {self.role}, using default")
            bids = [{"bid_id": 0, "price": 0, "quantity": 0}]
        else:
            bids = self.latest_bids
            self.latest_bids = None
            print(f"{self.role.capitalize()} strategy received {len(bids)} bids")

        # ---- BUILD ORDERBOOK ----
        orders = []

        start_time, end_time, only_hours = product_tuples[0]
        c_rate = market_config.param_dict['allowed_c_rates'][0]

        for bid in bids:
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

        print(f"{self.role.capitalize()} orderbook generated with {len(orders)} orders")
        return orders

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"{self.role.capitalize()} strategy connected to MQTT broker, rc={rc}")
        client.subscribe([(self.TOPIC_BIDS, 0)])

    def on_message(self, client, userdata, msg):
        if msg.topic == self.TOPIC_BIDS:
            payload = json.loads(msg.payload.decode())
            print(f"{self.role.capitalize()} strategy received bid(s): {payload}")

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
                json.dumps({
                    "ack": "bids_received",
                    "count": len(bids),
                    "bid_ids": [b.get("bid_id", 0) for b in bids if isinstance(b, dict)]
                })
            )

            print("Confirmed bids received")

    def calculate_reward(self, unit: BaseUnit, marketconfig: MarketConfig, orderbook: Orderbook):
        print('in calculate reward')
        self.market_open_sent = False

        payload = {
            "msg": "market result",
            "market_id": getattr(marketconfig, "market_id", None),
            "orderbook": orderbook,#.__dict__,
            "unit_id": getattr(unit, "id", None),
        }

        if hasattr(self, "client"):
            msg_info = self.client.publish(
                self.TOPIC_RESULTS,
                json.dumps(payload, default=str, indent=4),
                qos=1,
                retain=False
            )
            msg_info.wait_for_publish()
            print(f"Market reward sent via MQTT on {self.TOPIC_RESULTS} for unit {unit.id}")
            
            
# # SPDX-FileCopyrightText: MAS4TE Developers
# #
# # SPDX-License-Identifier: AGPL-3.0-or-later

# import time
# import json
# import paho.mqtt.client as mqtt
# from datetime import datetime
# from typing import List

# from assume.common.market_objects import Orderbook, Product, MarketConfig
# from assume.common.base import BaseStrategy, BaseUnit, SupportsMinMaxCharge


# class LLMStrategy(BaseStrategy):
#     """
#     MQTT-based strategy that can act as buyer or seller.
#     """
    
#     # Shared MQTT clients across all instances
#     _mqtt_clients = {}

#     def __init__(self, unit_id, role="buy", market_config=None, comm_agent_port=8000):
#         self.unit_id = unit_id
#         self.role = role  # "buy" or "sell"
#         self.market_config = market_config
#         self.latest_bid = None
#         self.market_open_sent = False

#         # MQTT topics
#         self.TOPIC_MARKET_STATUS = f"mas4te/market/status_agent{str(self.unit_id)}"
#         self.TOPIC_BIDS = f"mas4te/bids/agent{str(self.unit_id)}"
#         self.TOPIC_RESULTS = f"mas4te/results/agent{str(self.unit_id)}"

#         # MQTT setup - reuse client if already exists for this role/unit
#         client_key = f"{role}_{unit_id}_{comm_agent_port}"
#         if client_key not in LLMStrategy._mqtt_clients:
#             self.client = mqtt.Client(client_id=f"assume_{comm_agent_port}_{role}_{unit_id}")
#             self.client.on_connect = self.on_connect
#             self.client.on_message = self.on_message
#             self.client.connect("localhost", 1883, 60)
#             self.client.loop_start()
#             LLMStrategy._mqtt_clients[client_key] = self.client
#         else:
#             self.client = LLMStrategy._mqtt_clients[client_key]

#     def calculate_bids(self, unit, product_tuples: List[tuple], market_config=None, **kwargs):
#         """
#         Wait for a bid from the agent and return it as an Orderbook.
#         Each product tuple is (start_time: datetime, end_time: datetime, only_hours)
#         """
        
#         # Send market open message on first call (when we have products)
#         if not self.market_open_sent and self.market_config:
#             # Build products payload from actual product_tuples
#             products_payload = []
#             for p in product_tuples:
#                 start_time, end_time, only_hours = p
#                 products_payload.append({
#                     "start_time": start_time.isoformat() if hasattr(start_time, 'isoformat') else str(start_time),
#                     "end_time": end_time.isoformat() if hasattr(end_time, 'isoformat') else str(end_time),
#                     "only_hours": only_hours,
#                 })
            
#             # Build market products from config
#             market_products_payload = []
#             for p in getattr(self.market_config, "market_products", []):
#                 market_products_payload.append({
#                     "duration_seconds": p.duration.total_seconds(),
#                     "count": p.count,
#                     "first_delivery": p.first_delivery.total_seconds(),
#                     "only_hours": p.only_hours,
#                 })

#             market_open_msg = {
#                 "status": "market_open",
#                 "market_id": self.market_config.market_id,
#                 "product_type": self.market_config.product_type,
#                 "maximum_bid_volume": self.market_config.maximum_bid_volume,
#                 "maximum_bid_price": self.market_config.maximum_bid_price,
#                 "minimum_bid_price": self.market_config.minimum_bid_price,
#                 "volume_unit": self.market_config.volume_unit,
#                 "price_unit": self.market_config.price_unit,
#                 "additional_fields": self.market_config.additional_fields,
#                 "market_products": market_products_payload,
#                 "products": products_payload,
#             }

#             # Publish with retain flag
#             msg_info = self.client.publish(
#                 self.TOPIC_MARKET_STATUS,
#                 json.dumps(market_open_msg, indent=4),
#                 qos=1,
#                 retain=True
#             )
#             msg_info.wait_for_publish()  # Wait until published
#             self.market_open_sent = True
#             print(f"Market open message sent by {self.role} strategy")
#             time.sleep(0.2)  # Give MQTT time to propagate
        
#         # Now wait for bid from agent
#         print(f"{self.role.capitalize()} strategy waiting for bid from agent...")
#         timeout = 30000
#         start = time.time()
#         while self.latest_bid is None and (time.time() - start < timeout):
#             time.sleep(0.05)

#         if self.latest_bid is None:
#             bid = {"bid_id": 0, "price": 0, "quantity": 0}
#             print(f"No bid received within timeout for {self.role}, using default bid")
#         else:
#             bid = self.latest_bid
#             self.latest_bid = None  # reset for next timestep
#             print(f"{self.role.capitalize()} strategy received bid: {bid}")

#         start_time, end_time, only_hours = product_tuples[0]
#         # print(f"DEBUG - Product tuple: ({start_time}, {end_time}, {only_hours})")
        
#         vol = bid["quantity"]
        
#         if self.role == "buy":
#             vol = -vol  # positive for sell, negative for buy
       
#         # Always take the first allowed c-rate; currently this is always 1
#         c_rate = market_config.param_dict['allowed_c_rates'][0] 

#         # print("bid id ", int(bid['bid_id']))

#         order = {
#             "bid_id": bid['bid_id'],
#             "start_time": start_time,
#             "end_time": end_time,
#             "volume": vol,
#             # "accepted_volume": 0.0,
#             "price": bid["price"],
#             # "accepted_price": 0.0,
#             "agent_addr": getattr(unit, "addr", "agent_addr_unknown"),
#             "node": getattr(unit, "node", "node_unknown"),
#             "only_hours": only_hours if only_hours is not None else [],
#             "c_rate":c_rate,#1, first of allowed c-rates
#         }

#         print(f"{self.role.capitalize()} orderbook generated from bid_id {bid.get('bid_id', 0)}")
#         return [order]

#     def on_connect(self, client, userdata, flags, rc, properties=None):
#         """Subscribe to the bid topic when connected."""
#         print(f"{self.role.capitalize()} strategy connected to MQTT broker, rc={rc}")
#         client.subscribe([(self.TOPIC_BIDS, 0)])

#     def on_message(self, client, userdata, msg):
#         """Handle incoming bid messages from the agent."""
#         if msg.topic == self.TOPIC_BIDS:
#             bid = json.loads(msg.payload.decode())
#             print(f"{self.role.capitalize()} strategy received bid: {bid}")
#             self.latest_bid = bid

#             # Acknowledge back to agent
#             client.publish(
#                 self.TOPIC_RESULTS,
#                 json.dumps({
#                     "ack": "bid_received",
#                     "bid_id": bid.get("bid_id", 0),
#                     "price": bid.get("price", 0),
#                     "quantity": bid.get("quantity", 0)
#                 })
#             )
#             print("Confirm bid received")

#     def calculate_reward(
#         self,
#         unit: BaseUnit,
#         marketconfig: MarketConfig,
#         orderbook: Orderbook,
#     ):
#         """
#         Calculates the reward for the given unit.

#         Args:
#             unit (BaseUnit): The unit.
#             marketconfig (MarketConfig): The market configuration.
#             orderbook (Orderbook): The orderbook.
#         """

#         print('in calculate reward')
#         self.market_open_sent = False

#         payload = {
#             "msg": "market result",
#             "market_id": getattr(marketconfig, "market_id", None),
#             "orderbook": orderbook.__dict__,  # must be JSON serializable
#             "unit_id": getattr(unit, "id", None),
#         }


#         # Publish using the existing MQTT client and topic
#         if hasattr(self, "client") and hasattr(self, "TOPIC_RESULTS"):
#             msg_info = self.client.publish(
#                 self.TOPIC_RESULTS,
#                 json.dumps(payload, default=str, indent=4),
#                 qos=1,
#                 retain=False  # rewards are usually transient
#             )
#             msg_info.wait_for_publish()
#             print(f"Market reward sent via MQTT on {self.TOPIC_RESULTS} for unit {unit.id}")
#         else:
#             print("MQTT client or results topic not initialized, cannot send reward.")




# # import json
# # import time
# # import paho.mqtt.client as mqtt

# # from battery_utility_calculator import Storage
# # from assume.common.base import BaseStrategy, SupportsMinMaxCharge
# # from assume.common.market_objects import Orderbook, Product, MarketConfig
# # from assume.common.base import BaseStrategy, BaseUnit, SupportsMinMaxCharge



# # class LLMStrategy(BaseStrategy):
# #     """
# #     MQTT-based strategy that can act as buyer or seller.
# #     """

# #     def __init__(self, unit_id, role="buy", baseline_storage=0, market_config=None, comm_agent_port=8000):
# #         super().__init__()
# #         self.unit_id = unit_id
# #         self.role = role  # "buy" or "sell"
# #         self.baseline_storage = baseline_storage
# #         self.market_config = market_config
# #         self.latest_bid = None
# #         self.market_open_sent = False

# #         # MQTT topics
# #         self.TOPIC_MARKET_STATUS = f"mas4te/market/status_agent{str(self.unit_id)}"
# #         # self.TOPIC_MARKET_STATUS = "mas4te/market/status_agent01"

# #         self.TOPIC_BIDS = f"mas4te/bids/agent{str(self.unit_id)}"
# #         self.TOPIC_RESULTS = f"mas4te/results/agent{str(self.unit_id)}" 

# #         # MQTT setup
# #         self.client = mqtt.Client(client_id=f"assume_{comm_agent_port}")
# #         self.client.on_connect = self.on_connect
# #         self.client.on_message = self.on_message
# #         self.client.connect("localhost", 1883, 60)
# #         self.client.loop_start()

        
# #     def calculate_bids(self, unit: SupportsMinMaxCharge, product_tuples: list[Product], **kwargs) -> Orderbook:
# #         """Wait for a bid from the agent and return it as an Orderbook."""

# #         # if self.market_config is not None:
# #         #     # safe to use self.market_config
# #         #     print("Market config is available:", self.market_config)
# #         # else:
# #         #     print("No market config available")


# #         print()
# #         print(self.unit_id, self.TOPIC_MARKET_STATUS)
# #         print(self.unit_id, self.TOPIC_BIDS)

# #         product = product_tuples[0]
# #         print('product ', product)

# #         print(f"{self.role.capitalize()} strategy waiting for bid from agent...")
# #         timeout = 3000
# #         start = time.time()
# #         while self.latest_bid is None and (time.time() - start < timeout):
# #             time.sleep(0.05)

# #         if self.latest_bid is None:
# #             bid = {"bid_id": 0, "price": 0, "quantity": 0}
# #             print(f"No bid received within timeout for {self.role}, using default bid")
# #         else:
# #             print(f"{self.role}, update self.latest_bid")
# #             bid = self.latest_bid
# #             self.latest_bid = None  # reset for next timestep

# #             product = product_tuples[0]
# #             print('product ', product)


# #             price = bid['price']
# #             vol = bid['quantity']
# #             bid_id = bid['bid_id']
# #             if self.role == 'buy':
# #                 vol = - vol

# #             order = {
# #                 "bid_id": f"{unit.unit_id}_{i}",
# #                 "start_time": product.start_time,
# #                 "end_time": product.end_time,
# #                 "volume": vol,
# #                 "accepted_volume": 0.0,
# #                 "price": price,
# #                 "accepted_price": 0.0,
# #                 "agent_addr": unit.addr,
# #                 "node": unit.node,
# #                 "only_hours": product.only_hours,
# #                 }   


# #             orderbook = Orderbook(bids=[bid])
# #             print('orderbook ', orderbook)
# #             print(f"{self.role.capitalize()} orderbook generated from bid_id {bid.get('bid_id', 0)}")
# #             return orderbook

# #     def on_connect(self, client, product_tuples, userdata, flags, rc, properties=None):
# #         """Subscribe to the bid topic when connected."""
# #         print(f"{self.role.capitalize()} strategy connected to MQTT broker, rc={rc}")

# #         # client.subscribe((self.TOPIC_BIDS,0))
# #         client.subscribe([(self.TOPIC_BIDS, 0)])

# #         # Only send market open once
# #         if not self.market_open_sent and self.market_config:
# #             # market_open_msg = {"status": "market_open"}
# #             # market_open_msg = {
# #             #     "status": "market_open",
# #             #     "market_id": "test"
# #             # }
# #             products_payload=[]
# #             for p in product_tuples:
# #                 products_payload.append({
# #                     "start_time": p.start_time.isoformat(),
# #                     "end_time": p.end_time.isoformat(),
# #                     "duration_seconds": p.duration.total_seconds(),
# #                     "only_hours": p.only_hours,
# #                 })

# #             market_open_msg = {
# #                 "status": "market_open",
# #                 "market_id": self.market_config.market_id,
# #                 "product_type": self.market_config.product_type,
# #                 "maximum_bid_volume": self.market_config.maximum_bid_volume,
# #                 "maximum_bid_price": self.market_config.maximum_bid_price,
# #                 "minimum_bid_price": self.market_config.minimum_bid_price,
# #                 "volume_unit": self.market_config.volume_unit,
# #                 "price_unit": self.market_config.price_unit,
# #                 "additional_fields": self.market_config.additional_fields,
# #                 "market_products": [
# #                     {
# #                         "duration": p.duration.total_seconds(),
# #                         "count": p.count,
# #                         "first_delivery": p.first_delivery.total_seconds(),
# #                         "only_hours": p.only_hours,
# #                     }
# #                     for p in self.market_config.market_products
# #                 ],
# #                 "products": products_payload,
# #             }

# #             client.publish(
# #                 self.TOPIC_MARKET_STATUS,
# #                 json.dumps(market_open_msg, indent=4),
# #                 retain=True
# #             )
# #             self.market_open_sent = True
# #             print(f"Market open message sent by {self.role} strategy")

        

# #     def on_message(self, client, userdata, msg):
# #         """Handle incoming bid messages from the agent."""
# #         if msg.topic == self.TOPIC_BIDS:
# #             bid = json.loads(msg.payload.decode())
# #             print(f"{self.role.capitalize()} strategy received bid: {bid}")
# #             self.latest_bid = bid

# #             # Acknowledge back to agent
# #             client.publish(
# #                 self.TOPIC_RESULTS,
# #                 json.dumps({
# #                     "ack": "bid_received",
# #                     "bid_id": bid.get("bid_id", 0),
# #                     "price": bid.get("price", 0),
# #                     "quantity": bid.get("quantity", 0)
# #                 })
# #             )
# #         print('Confirm bid received')

# #     def calculate_reward(
# #         self,
# #         unit: BaseUnit,
# #         marketconfig: MarketConfig,
# #         orderbook: Orderbook,
# #     ):
# #         """
# #         Calculates the reward for the given unit.

# #         Args:
# #             unit (BaseUnit): The unit.
# #             marketconfig (MarketConfig): The market configuration.
# #             orderbook (Orderbook): The orderbook.
# #         """
# #         print('in calculate reward')
# #         # self.market_to_llm_queue.put(
# #         #     {
# #         #         "msg": "market result",
# #         #         # "market_config": marketconfig,
# #         #         "orderbook": orderbook,
# #         #     }
# #         # )






# # import json
# # import time
# # import paho.mqtt.client as mqtt

# # from battery_utility_calculator import Storage
# # from assume.common.base import BaseStrategy, SupportsMinMaxCharge
# # from assume.common.market_objects import Orderbook, Product, MarketConfig


# # class LLMStrategy(BaseStrategy):
# #     """
# #     Base strategy for a storage unit interacting with an agent via MQTT.
# #     """

# #     def __init__(self, baseline_storage=0, market_config = None, comm_agent_port=8000, *args, **kwargs):
# #         super().__init__()
# #         self.baseline_storage = baseline_storage
# #         self.latest_bid = None
# #         self.market_config = market_config
# #         self.market_open_published = False

# #         # MQTT setup
# #         self.client = mqtt.Client(client_id=f"assume_{comm_agent_port}")
# #         self.client.on_connect = self.on_connect
# #         self.client.on_message = self.on_message
# #         self.client.connect("localhost", 1883, 60)
# #         self.client.loop_start()  # run MQTT in background

# #         # Topics
# #         self.TOPIC_MARKET_STATUS = "mas4te/market/status"
# #         self.TOPIC_BIDS = f"mas4te/bids/agent{comm_agent_port - 8000 + 1}"
# #         self.TOPIC_RESULTS = f"mas4te/results/agent{comm_agent_port - 8000 + 1}"

# #     def on_connect(self, client, userdata, flags, rc, properties=None):
# #         print(f"ASSUME connected to MQTT broker, rc={rc}")
# #         client.subscribe(self.TOPIC_BIDS)
# #         # if not self.market_open_published:

# #         #     print('market configuration before opening')
# #         #     print(self.market_config)

# #         #     client.publish(self.TOPIC_MARKET_STATUS, json.dumps({"status": "market_open"}), retain=True)
# #         #     self.market_open_published = True
# #         print("ASSUME -> connecte")

# #     def on_message(self, client, userdata, msg):
# #         bid = json.loads(msg.payload.decode())
# #         print(f"ASSUME received bid: {bid}")
# #         self.latest_bid = bid

# #         # Confirm bid back to agent
# #         client.publish(self.TOPIC_RESULTS, json.dumps({
# #             "ack": "bid_received",
# #             "bid_id": bid.get("bid_id", 0),
# #             "price": bid.get("price", 0),
# #             "quantity": bid.get("quantity", 0)
# #         }))
# #         print(f"ASSUME -> bid_received published for bid_id {bid.get('bid_id', 0)}")

# #     def open_market(self):
# #         print(self.market_config)
# #         msg = {
# #             "status": "market_open", 
# #             "products": [p.__dict__ for p in self]
# #         }


# # class LLMBuyStrategy(LLMStrategy):
# #     """Strategy for storage buyers using the LLM + MQTT setup."""

# #     def calculate_bids(self, unit: SupportsMinMaxCharge, market_config: MarketConfig,
# #                        product_tuples: list[Product], **kwargs) -> Orderbook:
# #         """
# #         Waits for a bid from the agent and returns it as an Orderbook.
# #         """
# #         print("Waiting for bid from agent...")
# #         timeout = 2300  # seconds
# #         start = time.time()
# #         while self.latest_bid is None and (time.time() - start < timeout):
# #             time.sleep(0.05)

# #         if self.latest_bid is None:
# #             print("No bid received within timeout, using default bid.")
# #             bid = {"bid_id": 0, "price": 0, "quantity": 0}
# #         else:
# #             bid = self.latest_bid
# #             self.latest_bid = None  # reset for next market step

# #         orderbook = Orderbook(bids=[bid])
# #         print(f"Orderbook generated from bid_id {bid.get('bid_id', 0)}")
# #         return orderbook


# # class LLMSellStrategy(LLMStrategy):
# #     """Strategy for storage sellers using the LLM + MQTT setup."""

# #     def calculate_bids(self, unit: SupportsMinMaxCharge, market_config: MarketConfig,
# #                        product_tuples: list[Product], **kwargs) -> Orderbook:
# #         """
# #         Waits for a bid from the agent and returns it as an Orderbook.
# #         """
# #         print("Waiting for bid from agent...")
# #         timeout = 30
# #         start = time.time()
# #         while self.latest_bid is None and (time.time() - start < timeout):
# #             time.sleep(0.05)

# #         if self.latest_bid is None:
# #             print("No bid received within timeout, using default bid.")
# #             bid = {"bid_id": 0, "price": 0, "quantity": 0}
# #         else:
# #             bid = self.latest_bid
# #             self.latest_bid = None

# #         orderbook = Orderbook(bids=[bid])
# #         print(f"Orderbook generated from bid_id {bid.get('bid_id', 0)}")
# #         return orderbook




# # # SPDX-FileCopyrightText: MAS4TE Developers
# # #
# # # SPDX-License-Identifier: AGPL-3.0-or-later

# # from multiprocessing import Process, Queue

# # # import communication_agent
# # import requests
# # from battery_utility_calculator import Storage

# # from assume.common.base import BaseStrategy, BaseUnit, SupportsMinMaxCharge
# # from assume.common.market_objects import MarketConfig, Orderbook, Product


# # class LLMStrategy(BaseStrategy):
# #     """
# #     A strategy that uses a Large Language Model (LLM) for a storage buyer.

# #     Params:
# #         llm_api_url (str): The URL of the LLM API to use for generating bids.
# #     """

# #     def __init__(self, llm_api_url=None, baseline_storage=0, *args, **kwargs):
# #         super().__init__()
# #         self.baseline_storage = baseline_storage
# #         self.api_url = llm_api_url
# #         self.headers = {"Content-Type": "application/json"}
# #         self.storages_to_calculate = self.build_storages_to_calculate()

# #         self.market_to_llm_queue = Queue()
# #         self.llm_to_market_queue = Queue()

# #         self.process = Process(
# #             target=communication_agent.run_app,
# #             daemon=True,
# #             kwargs={
# #                 "port": kwargs.get("comm_agent_port", 8000),
# #                 "market_to_llm_queue": self.market_to_llm_queue,
# #                 "llm_to_market_queue": self.llm_to_market_queue,
# #             },
# #         )
# #         self.process.start()

# #     def build_storages_to_calculate(self):
# #         """Builds a list of storage volumes to calculate worth for.

# #         Returns:
# #             list[Storage]: List of Storage objects with different volumes.
# #         """
# #         # Example: Create storages with volumes from 0 to 1000 in steps of 100
# #         storages = [
# #             Storage(id=i, volume=i, c_rate=1, efficiency=0.95) for i in range(1, 15)
# #         ]

# #         storages += [
# #             Storage(id=i, volume=i * 5, c_rate=1, efficiency=0.95) for i in range(3, 11)
# #         ]

# #         return storages

# #     def run_prompt(
# #         self, prompt: str, model="Mistral-7B-Instruct-v0.3-Q4_K_M", max_tokens=1000
# #     ):
# #         data = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
# #         response = requests.post(self.api_url, headers=self.headers, json=data)
# #         response.raise_for_status()
# #         result = response.json()
# #         return result.get("choices", [{}])[0].get("text", "")


# # class LLMBuyStrategy(LLMStrategy):
# #     """A strategy that uses a Large Language Model (LLM) for a storage buyer."""

# #     def __init__(self, llm_api_url=None, baseline_storage=0, *args, **kwargs):
# #         super().__init__(llm_api_url, baseline_storage, *args, **kwargs)

# #     def calculate_bids(
# #         self,
# #         unit: SupportsMinMaxCharge,
# #         market_config: MarketConfig,
# #         product_tuples: list[Product],
# #         **kwargs,
# #     ) -> Orderbook:
# #         """Calculates the value of multiple storage volumes for a predicted demand and price timeseries via linear optimization.

# #         Args:
# #             unit (SupportsMinMaxCharge): The unit to calculate bids for.
# #             market_config (MarketConfig): The market configuration to use.
# #             product_tuples (list[Product]): The list of products to calculate bids for.

# #         Returns:
# #             Orderbook: The calculated order book with bids.
# #         """

# #         self.market_to_llm_queue.put(
# #             {
# #                 "msg": "calculate bids",
# #                 # "market_config": market_config,
# #                 "product_tuples": product_tuples[0],
# #             }
# #         )

# #         bids = self.llm_to_market_queue.get()

# #         return bids

# #     def calculate_reward(
# #         self,
# #         unit: BaseUnit,
# #         marketconfig: MarketConfig,
# #         orderbook: Orderbook,
# #     ):
# #         """
# #         Calculates the reward for the given unit.

# #         Args:
# #             unit (BaseUnit): The unit.
# #             marketconfig (MarketConfig): The market configuration.
# #             orderbook (Orderbook): The orderbook.
# #         """

# #         self.market_to_llm_queue.put(
# #             {
# #                 "msg": "market result",
# #                 # "market_config": marketconfig,
# #                 "orderbook": orderbook,
# #             }
# #         )


# # class LLMSellStrategy(LLMStrategy):
# #     """A strategy that uses a Large Language Model (LLM) for a storage seller."""

# #     def __init__(self, llm_api_url=None, baseline_storage=0, *args, **kwargs):
# #         super().__init__(llm_api_url, baseline_storage, *args, **kwargs)

# #     def calculate_bids(
# #         self,
# #         unit: SupportsMinMaxCharge,
# #         market_config: MarketConfig,
# #         product_tuples: list[Product],
# #         **kwargs,
# #     ) -> Orderbook:
# #         """Calculates the value of multiple storage volumes for a predicted demand and price timeseries via linear optimization.

# #         Args:
# #             unit (SupportsMinMaxCharge): The unit to calculate bids for.
# #             market_config (MarketConfig): The market configuration to use.
# #             product_tuples (list[Product]): The list of products to calculate bids for.

# #         Returns:
# #             Orderbook: The calculated order book with bids.
# #         """

# #         self.market_to_llm_queue.put(
# #             {
# #                 "msg": "calculate bids",
# #                 # "market_config": market_config,
# #                 "product_tuples": product_tuples[0],
# #             }
# #         )

# #         bids = self.llm_to_market_queue.get()

# #         return bids

# #     def calculate_reward(
# #         self,
# #         unit: BaseUnit,
# #         marketconfig: MarketConfig,
# #         orderbook: Orderbook,
# #     ):
# #         """
# #         Calculates the reward for the given unit.

# #         Args:
# #             unit (BaseUnit): The unit.
# #             marketconfig (MarketConfig): The market configuration.
# #             orderbook (Orderbook): The orderbook.
# #         """

# #         self.market_to_llm_queue.put(
# #             {
# #                 "msg": "market result",
# #                 # "market_config": marketconfig,
# #                 "orderbook": orderbook,
# #             }
# #         )
