# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from mqtt_agent_strategy import MQTTAgentStrategy


class BatteryMarketStrategy(MQTTAgentStrategy):
    """
    Bids of the AI agent on the battery (storage) market.

    Topics: ``mas4te/market/status_agent<id>``, ``mas4te/bids/agent<id>``,
    ``mas4te/results/agent<id>``.

    Bid format: ``{"bid_id", "price", "quantity", "blockchain_address"}``.
    ``quantity`` is positive, the sign comes from the role of the unit
    (``volume`` is accepted as fallback).
    """

    TOPIC_PREFIX = "mas4te"

    def make_orders(self, unit, bids, product_tuples, market_config):
        # one product per auction: the storage for the next period
        product = product_tuples[0]
        c_rate = market_config.param_dict["allowed_c_rates"][0]

        orders = []
        for bid in bids:
            if not isinstance(bid, dict):
                print(f"Skipping invalid bid: {bid}")
                continue

            # the agent is assigned its blockchain address and sends it with every bid
            if not bid.get("blockchain_address"):
                print(f"Skipping bid without blockchain_address: {bid}")
                continue

            # fallback if still using "volume"
            quantity = bid.get("quantity", bid.get("volume", 0))
            volume = quantity if self.role == "sell" else -quantity

            order = self.base_order(unit, bid, product, volume)
            order["c_rate"] = c_rate
            order["blockchain_address"] = bid["blockchain_address"]
            orders.append(order)

        return orders
