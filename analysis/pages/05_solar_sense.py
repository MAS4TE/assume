# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")

DB_URI = "postgresql://assume:assume@10.26.5.92:5432/assume"
sim_configs = pd.read_sql("SELECT * FROM sim_config", DB_URI)

st.header("Sensitivity analysis for 5kWh storages")
clearing_prices = pd.read_sql(
    sql="""
    select
        max(accepted_price) * 100 as clearing_price,
        start_time,
        simulation
    from market_orders
    where simulation like %(pattern)s
    group by start_time, simulation""",
    con=DB_URI,
    params={"pattern": "%solar%"},
)
prices_over_time = px.line(
    clearing_prices, x="start_time", y="clearing_price", color="simulation"
)
st.plotly_chart(prices_over_time)

traded_volumes = pd.read_sql(
    sql="""
    select
        sum(accepted_volume) as traded_vol,
        start_time,
        simulation
    from market_orders
    where simulation like %(pattern)s
    and accepted_volume > 0
    group by start_time, simulation""",
    con=DB_URI,
    params={"pattern": "%solar%"},
)
volumes_over_time = px.line(
    traded_volumes, x="start_time", y="traded_vol", color="simulation"
)
st.plotly_chart(volumes_over_time)
