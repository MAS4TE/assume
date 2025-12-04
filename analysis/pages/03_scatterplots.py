# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")

DB_URI = "postgresql://assume:assume@10.26.5.92:5432/assume"
SUPPLY_IDS = [
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
]


def get_solar_gen(sim_type):
    sql = """
    WITH starts AS (
        SELECT DISTINCT mo.start_time, mo.simulation, UNNEST (sc.choosen_demand_units) AS demand_units
        FROM market_orders AS mo
        LEFT JOIN sim_config AS sc ON sc.simulation = mo.simulation
        WHERE mo.simulation LIKE %s
        order by mo.start_time
    )
    SELECT
        starts.start_time,
        starts.simulation,
        sum(so.solar_gen_kw) / 4 AS solar_gen_kwh
    FROM
        simulation_data.solar_gen AS so
    RIGHT JOIN starts
    ON so.profile_id = starts.demand_units
    AND so.datetime >= starts.start_time
    AND so.datetime < starts.start_time + INTERVAL '7 days'
    GROUP BY start_time, simulation
"""
    if f"{sim_type}_solar" not in st.session_state:
        with st.spinner("Loading solar gens..."):
            st.session_state[f"{sim_type}_solar"] = pd.read_sql(
                sql, DB_URI, params=(f"%{sim_type}%",)
            )
            # solar_gens = pd.read_hdf(f"./analysis/data/{sim_type}_solar.hdf")
    solar_gens = st.session_state[f"{sim_type}_solar"].copy()
    return solar_gens


def get_wholesale_prices():
    sql = """
    WITH starts AS (
        SELECT DISTINCT start_time, simulation
        FROM market_orders
        WHERE simulation LIKE %s
    )
    SELECT
        starts.start_time,
        starts.simulation,
        STDDEV(wholesale) as wholesale_price
    FROM
        starts
    JOIN
        simulation_data.prices
    ON
        prices.datetime >= starts.start_time
    AND
        prices.datetime < (starts.start_time + interval '7 days')
    GROUP BY
        starts.start_time, starts.simulation
    ORDER BY
        starts.start_time
"""
    if "wholesale_prices" not in st.session_state:
        with st.spinner("Loading wholesale prices..."):
            st.session_state["wholesale_prices"] = pd.read_sql(
                sql, DB_URI, params=("%base%",)
            )
            # wholesale_prices = pd.read_hdf("./analysis/data/wholesale_prices.hdf")
    wholesale_prices = st.session_state["wholesale_prices"].copy()
    return wholesale_prices


def get_clearing_prices(sim_type):
    sql = """
    SELECT
        start_time,
        simulation,
        AVG(accepted_price) as clearing_price
    FROM
        market_orders
    WHERE
        accepted_volume != 0
    AND
        simulation LIKE %s
    GROUP BY
        start_time, simulation
    ORDER BY
        start_time
"""
    if f"{sim_type}_clearing" not in st.session_state:
        with st.spinner("Loading clearing prices..."):
            st.session_state[f"{sim_type}_clearing"] = pd.read_sql(
                sql, DB_URI, params=(f"%{sim_type}%",)
            )
            # clearing_prices = pd.read_hdf(f"./data/{sim_type}_clearing_prices.hdf")
    clearing_prices = st.session_state[f"{sim_type}_clearing"].copy()
    return clearing_prices


# use_all_sims = st.toggle("Use all simulations", value=True)
use_all_sims = True
with st.spinner("Loading sim configs..."):
    sim_configs = pd.read_sql("SELECT * FROM sim_config", DB_URI)
sim_ids = sim_configs["simulation"].sort_values().unique().tolist()
# sim_id = st.selectbox(label="Simulation ID", options=sim_ids)

sql = """
    SELECT
        choosen_supply_units as su,
        choosen_demand_units as du
    FROM
        sim_config
    WHERE
        simulation LIKE %s
"""
# with st.spinner("Loading units..."):
#     unit_ids = pd.read_sql(sql, DB_URI, params=("%base%"))
#     supply_ids = [int(id) for id in unit_ids.loc[0, "su"].replace("{", "").replace("}", "").split(",")]
#     demand_ids = [int(id) for id in unit_ids.loc[0, "du"].replace("{", "").replace("}", "").split(",")]
demand_ids = list(range(120))
supply_ids = SUPPLY_IDS


wholesale_prices = get_wholesale_prices()
###################
# BASE SIMULATION #
###################
st.header("Base simulation - 50 demand & 50 supply units with 5kWh storage")
base_solar_gens = get_solar_gen(sim_type="base")
base_clearing_prices = get_clearing_prices(sim_type="base")

base_df = pd.merge(
    base_clearing_prices, base_solar_gens, on=["start_time", "simulation"]
).merge(wholesale_prices, on="start_time", how="left")
# st.dataframe(base_df)
base_fig = px.scatter(
    base_df,
    x="wholesale_price",
    y="clearing_price",
    color="solar_gen_kwh",
    color_continuous_scale="Turbo",
    trendline="ols",
)
base_fig_3d = px.scatter_3d(
    base_df,
    x="wholesale_price",
    y="clearing_price",
    z="solar_gen_kwh",
    color="solar_gen_kwh",
    color_continuous_scale="Turbo",
)
base_fig_3d.update_traces(
    marker={"size": 2},
)
st.plotly_chart(base_fig, key="basefig")
st.plotly_chart(base_fig_3d, key="basefig_3d")

##################
# SUPPLY SURPLUS #
##################
st.header("Supply surplus simulation - 50 demand & 50 supply units with 10kWh storage")
supply_surplus_solar_gens = get_solar_gen(sim_type="supply")
supply_surplus_clearing_prices = get_clearing_prices(sim_type="supply")

supply_surplus_df = pd.merge(
    supply_surplus_clearing_prices,
    supply_surplus_solar_gens,
    on=["start_time", "simulation"],
).merge(wholesale_prices, on="start_time", how="left")
# st.dataframe(supply_surplus_df)
supply_surplus_fig = px.scatter(
    supply_surplus_df,
    x="wholesale_price",
    y="clearing_price",
    color="solar_gen_kwh",
    color_continuous_scale="Turbo",
)
supply_surplus_fig_3d = px.scatter_3d(
    supply_surplus_df,
    x="wholesale_price",
    y="clearing_price",
    z="solar_gen_kwh",
    color="solar_gen_kwh",
    color_continuous_scale="Turbo",
)
supply_surplus_fig_3d.update_traces(
    marker={"size": 2},
)
st.plotly_chart(supply_surplus_fig, key="ssfig")
st.plotly_chart(supply_surplus_fig_3d, key="ssfig_3d")

##################
# DEMAND SURPLUS #
##################
st.header("Demand surplus simulation - 50 demand & 50 supply units with 2.5kWh storage")
demand_surplus_solar_gens = get_solar_gen(sim_type="demand")
demand_surplus_clearing_prices = get_clearing_prices(sim_type="demand")

demand_surplus_df = pd.merge(
    demand_surplus_clearing_prices,
    demand_surplus_solar_gens,
    on=["start_time", "simulation"],
).merge(wholesale_prices, on="start_time", how="left")
# st.dataframe(demand_surplus_df)
demand_surplus_fig = px.scatter(
    demand_surplus_df,
    x="wholesale_price",
    y="clearing_price",
    color="solar_gen_kwh",
    color_continuous_scale="Turbo",
)
demand_surplus_fig_3d = px.scatter_3d(
    demand_surplus_df,
    x="wholesale_price",
    y="clearing_price",
    z="solar_gen_kwh",
    color="solar_gen_kwh",
    color_continuous_scale="Turbo",
)
demand_surplus_fig_3d.update_traces(
    marker={"size": 2},
)
st.plotly_chart(demand_surplus_fig, key="dsfig")
st.plotly_chart(demand_surplus_fig_3d, key="dsfig_3d")
