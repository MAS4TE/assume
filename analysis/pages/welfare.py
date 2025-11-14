# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")

DB_URI = "postgresql://assume:assume@10.26.5.92:5432/assume"
sim_configs = pd.read_sql("SELECT * FROM sim_config", DB_URI)

###################
# BASE SIMULATION #
###################
st.header("Base simulation - 50 demand & 50 supply units with 5kWh storage")
base_total_welfare = pd.read_sql(
    sql="""
        SELECT
            SUM(ABS(price - accepted_price)) AS welfare,
            simulation,
            start_time
        FROM market_orders
        WHERE simulation LIKE %(pattern)s
        GROUP BY simulation, start_time
    """,
    con=DB_URI,
    params={"pattern": "%base%"},
)

base_total_welfare_line = px.line(
    base_total_welfare, "start_time", "welfare", color="simulation"
)
base_total_welfare_line.update_layout(
    title="Total welfare over time",
    xaxis_title="Date",
    yaxis_title="Welfare in €",
)
st.plotly_chart(base_total_welfare_line)

base_total_welfare_box = px.box(base_total_welfare, "welfare", color="simulation")
base_total_welfare_box.update_layout(title="Total welfare", xaxis_title="Welfare in €")
st.plotly_chart(base_total_welfare_box)

base_seller_welfare = pd.read_sql(
    sql="""
        select
            sum(accepted_price - price) AS welfare,
            start_time,
            simulation
        from market_orders
        where simulation like %(pattern)s
        and accepted_volume > 0
        group by start_time, simulation
    """,
    con=DB_URI,
    params={"pattern": "%base%"},
)
base_seller_welfare_box = px.box(base_seller_welfare, "welfare", color="simulation")
base_seller_welfare_box.update_layout(
    title="Sellers welfare", xaxis_title="Welfare in €"
)
st.plotly_chart(base_seller_welfare_box)

base_buyer_welfare = pd.read_sql(
    sql="""
        select
            sum(price - accepted_price) AS welfare,
            start_time,
            simulation
        from market_orders
        where simulation like %(pattern)s
        and accepted_volume < 0
        group by start_time, simulation
    """,
    con=DB_URI,
    params={"pattern": "%base%"},
)
base_buyer_welfare_box = px.box(base_buyer_welfare, "welfare", color="simulation")
base_buyer_welfare_box.update_layout(title="Buyers welfare", xaxis_title="Welfare in €")
st.plotly_chart(base_buyer_welfare_box)

##################
# SUPPLY SURPLUS #
##################
st.header("Supply surplus simulation - 50 demand & 50 supply units with 10kWh storage")
supply_total_welfare = pd.read_sql(
    sql="""
        SELECT
            SUM(ABS(price - accepted_price)) AS welfare,
            simulation,
            start_time
        FROM market_orders
        WHERE simulation LIKE %(pattern)s
        AND accepted_volume != 0
        GROUP BY simulation, start_time
    """,
    con=DB_URI,
    params={"pattern": "%base%"},
)
supply_total_welfare_line = px.line(
    supply_total_welfare, "start_time", "welfare", color="simulation"
)
supply_total_welfare_line.update_layout(
    title="Total welfare over time",
    xaxis_title="Date",
    yaxis_title="Welfare in €",
)
st.plotly_chart(supply_total_welfare_line)

supply_total_welfare_box = px.box(supply_total_welfare, "welfare", color="simulation")
supply_total_welfare_box.update_layout(
    title="Total welfare", xaxis_title="Welfare in €"
)
st.plotly_chart(supply_total_welfare_box)

supply_seller_welfare = pd.read_sql(
    sql="""
        select
            sum(accepted_price - price) AS welfare,
            start_time,
            simulation
        from market_orders
        where simulation like %(pattern)s
        and accepted_volume > 0
        group by start_time, simulation
    """,
    con=DB_URI,
    params={"pattern": "%supply%"},
)
supply_seller_welfare_box = px.box(supply_seller_welfare, "welfare", color="simulation")
supply_seller_welfare_box.update_layout(
    title="Sellers welfare", xaxis_title="Welfare in €"
)
st.plotly_chart(supply_seller_welfare_box)

supply_buyer_welfare = pd.read_sql(
    sql="""
        select
            sum(price - accepted_price) AS welfare,
            start_time,
            simulation
        from market_orders
        where simulation like %(pattern)s
        and accepted_volume < 0
        group by start_time, simulation
    """,
    con=DB_URI,
    params={"pattern": "%supply%"},
)
supply_buyer_welfare_box = px.box(supply_buyer_welfare, "welfare", color="simulation")
supply_buyer_welfare_box.update_layout(
    title="Buyers welfare", xaxis_title="Welfare in €"
)
st.plotly_chart(supply_buyer_welfare_box)

##################
# DEMAND SURPLUS #
##################
st.header("Demand surplus simulation - 50 demand & 50 supply units with 2.5kWh storage")
demand_total_welfare = pd.read_sql(
    sql="""
        SELECT
            SUM(ABS(price - accepted_price)) AS welfare,
            simulation,
            start_time
        FROM market_orders
        WHERE simulation LIKE %(pattern)s
        GROUP BY simulation, start_time
    """,
    con=DB_URI,
    params={"pattern": "%base%"},
)

demand_total_welfare_line = px.line(
    demand_total_welfare, "start_time", "welfare", color="simulation"
)
demand_total_welfare_line.update_layout(
    title="Total welfare over time",
    xaxis_title="Date",
    yaxis_title="Welfare in €",
)
st.plotly_chart(demand_total_welfare_line)

demand_total_welfare_box = px.box(demand_total_welfare, "welfare", color="simulation")
demand_total_welfare_box.update_layout(
    title="Total welfare", xaxis_title="Welfare in €"
)
st.plotly_chart(demand_total_welfare_box)

demand_seller_welfare = pd.read_sql(
    sql="""
        select
            sum(accepted_price - price) welfare,
            start_time,
            simulation
        from market_orders
        where simulation like %(pattern)s
        and accepted_volume > 0
        group by start_time, simulation
    """,
    con=DB_URI,
    params={"pattern": "%demand%"},
)
demand_seller_welfare_box = px.box(demand_seller_welfare, "welfare", color="simulation")
demand_seller_welfare_box.update_layout(
    title="Sellers welfare", xaxis_title="Welfare in €"
)
st.plotly_chart(demand_seller_welfare_box)

demand_buyer_welfare = pd.read_sql(
    sql="""
        select
            sum(price - accepted_price) AS welfare,
            start_time,
            simulation
        from market_orders
        where simulation like %(pattern)s
        and accepted_volume < 0
        group by start_time, simulation
    """,
    con=DB_URI,
    params={"pattern": "%demand%"},
)
demand_buyer_welfare_box = px.box(demand_buyer_welfare, "welfare", color="simulation")
demand_buyer_welfare_box.update_layout(
    title="Buyers welfare", xaxis_title="Welfare in €"
)
st.plotly_chart(demand_buyer_welfare_box)
