# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
from mqtt_agent_strategy import MQTTAgentStrategy


class CommunityEOMStrategy(MQTTAgentStrategy):
    """
    Bids of the AI agent on the community energy-only market (CommunityEOM).

    Topics: ``mas4te/community_eom/market/status_agent<id>``,
    ``mas4te/community_eom/bids/agent<id>``, ``mas4te/community_eom/results/agent<id>``.

    One auction offers many delivery periods (e.g. 96 quarter hours), so every
    bid names its period with ``start_time`` (may be left out if only one is
    open). Prosumers can buy and sell on the same market, so the direction comes
    from the bid: ``side`` ``"sell"`` or ``"buy"``, or a signed ``volume``
    (> 0 sell, < 0 buy). Without either, ``quantity`` is signed by the role of
    the unit. Fields listed in ``additional_fields`` of the market (e.g. the
    blockchain address) are required.
    """

    TOPIC_PREFIX = "mas4te/community_eom"

    def make_orders(self, unit, bids, product_tuples, market_config):
        products = {pd.Timestamp(product[0]): product for product in product_tuples}
        additional_fields = getattr(market_config, "additional_fields", None) or []

        orders = []
        for bid in bids:
            if not isinstance(bid, dict):
                print(f"Skipping invalid bid: {bid}")
                continue

            if "start_time" in bid:
                product = products.get(pd.Timestamp(bid["start_time"]))
            elif len(products) == 1:
                product = next(iter(products.values()))
            else:
                product = None
            if product is None:
                print(f"Skipping bid for a period that is not open: {bid}")
                continue

            missing = [field for field in additional_fields if not bid.get(field)]
            if missing:
                print(f"Skipping bid without {', '.join(missing)}: {bid}")
                continue

            if bid.get("side") in ("sell", "buy"):
                volume = abs(bid.get("quantity", bid.get("volume", 0)))
                volume = volume if bid["side"] == "sell" else -volume
            elif "volume" in bid:
                volume = bid["volume"]
            else:
                quantity = bid.get("quantity", 0)
                volume = quantity if self.role == "sell" else -quantity
            if not volume:
                continue

            order = self.base_order(unit, bid, product, volume)
            for field in additional_fields:
                order[field] = bid[field]
            orders.append(order)

        return orders
