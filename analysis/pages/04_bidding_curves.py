# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")

DB_URI = "postgresql://assume:assume@10.26.5.92:5432/assume"
sim_configs = pd.read_sql("SELECT * FROM sim_config", DB_URI).sort_values("simulation")
start_times = pd.read_sql(
    "SELECT distinct(start_time) FROM market_orders", DB_URI
).sort_values("start_time")

lcol, rcol = st.columns(2)
sim_id = lcol.selectbox(label="Sim ID", options=sim_configs["simulation"].unique())
market_opening = rcol.selectbox(label="Market opening", options=start_times)

sql = f"""
    SELECT
        price,
        volume
    FROM market_orders
    WHERE simulation = '{sim_id}'
    AND start_time = '{market_opening}'
"""
sell_orders = (
    pd.read_sql(sql + " AND volume > 0", DB_URI)
    .sort_values("price")
    .reset_index(drop=True)
)
buy_orders = (
    pd.read_sql(sql + " AND volume < 0", DB_URI)
    .sort_values("price", ascending=False)
    .reset_index(drop=True)
)
buy_orders["volume"] = buy_orders["volume"].abs()
sell_orders["cum_vol"] = sell_orders["volume"].cumsum()
buy_orders["cum_vol"] = buy_orders["volume"].cumsum()

fig = go.Figure()
fig.add_scatter(x=sell_orders["cum_vol"], y=sell_orders["price"], name="Supply")
fig.add_scatter(x=buy_orders["cum_vol"], y=buy_orders["price"], name="Demand")
st.plotly_chart(fig)
