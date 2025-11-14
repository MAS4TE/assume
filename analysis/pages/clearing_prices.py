# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")

DB_URI = "postgresql://assume:assume@10.26.5.92:5432/assume"
sim_configs = pd.read_sql("SELECT * FROM sim_config", DB_URI)

st.header("Base simulation - 50 demand & 50 supply units with 5kWh storage")
base_clearing_prices = pd.read_sql(
    sql="""
    select
        max(accepted_price) * 100 as clearing_price,
        start_time,
        simulation
    from market_orders
    where simulation like %(pattern)s
    group by start_time, simulation""",
    con=DB_URI,
    params={"pattern": "%base%"},
)

base_clearing_price_box = px.box(
    base_clearing_prices, "clearing_price", color="simulation"
)
base_clearing_price_box.update_layout(
    title="Clearing prices over whole simulation",
    xaxis_title="Price in ct./kWh",
    yaxis_title="Simulation",
    xaxis_range=[0, 170],
)
st.plotly_chart(base_clearing_price_box)

base_clearing_price_line = px.line(
    base_clearing_prices,
    "start_time",
    "clearing_price",
    color="simulation",
    markers="+",
)
base_clearing_price_line.update_layout(
    title="Clearing prices over time",
    xaxis_title="Date",
    yaxis_title="Clearing price in ct./kWh",
)
st.plotly_chart(base_clearing_price_line)

st.header("Supply surplus simulation - 50 demand & 50 supply units with 10kWh storage")
supply_sur_clearing_prices = pd.read_sql(
    sql="""
    select
        max(accepted_price) * 100 as clearing_price,
        start_time,
        simulation
    from market_orders
    where simulation like %(pattern)s
    group by start_time, simulation""",
    con=DB_URI,
    params={"pattern": "%supply%"},
)

supply_sur_clearing_price_box = px.box(
    supply_sur_clearing_prices, "clearing_price", color="simulation"
)
supply_sur_clearing_price_box.update_layout(
    title="Clearing prices over whole simulation",
    xaxis_title="Price in ct./kWh",
    yaxis_title="Simulation",
    xaxis_range=[0, 170],
)
st.plotly_chart(supply_sur_clearing_price_box)

supply_sur_clearing_price_line = px.line(
    supply_sur_clearing_prices,
    "start_time",
    "clearing_price",
    color="simulation",
    markers="+",
)
supply_sur_clearing_price_line.update_layout(
    title="Clearing prices over time",
    xaxis_title="Date",
    yaxis_title="Clearing price in ct./kWh",
)
st.plotly_chart(supply_sur_clearing_price_line)

st.header("Demand surplus simulation - 50 demand & 50 supply units with 2.5kWh storage")
demand_sur_clearing_prices = pd.read_sql(
    sql="""
    select
        max(accepted_price) * 100 as clearing_price,
        start_time,
        simulation
    from market_orders
    where simulation like %(pattern)s
    group by start_time, simulation""",
    con=DB_URI,
    params={"pattern": "%demand%"},
)

demand_sur_clearing_price_box = px.box(
    demand_sur_clearing_prices, "clearing_price", color="simulation"
)
demand_sur_clearing_price_box.update_layout(
    title="Clearing prices over whole simulation",
    xaxis_title="Price in ct./kWh",
    yaxis_title="Simulation",
    xaxis_range=[0, 170],
)
st.plotly_chart(demand_sur_clearing_price_box)

demand_sur_clearing_price_line = px.line(
    demand_sur_clearing_prices,
    "start_time",
    "clearing_price",
    color="simulation",
    markers="+",
)
demand_sur_clearing_price_line.update_layout(
    title="Clearing prices over time",
    xaxis_title="Date",
    yaxis_title="Clearing price in ct./kWh",
)
st.plotly_chart(demand_sur_clearing_price_line)

##
# fig = go.Figure(base_clearing_price_line.data + supply_sur_clearing_price_line.data)
# st.plotly_chart(fig)
