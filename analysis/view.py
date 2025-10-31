# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st

st.set_page_config(layout="wide")

DB_URI = "postgresql://assume:assume@localhost:5432/assume"
con = psycopg2.connect(DB_URI)

# if "available_sims" not in st.session_state:
available_sims = pd.read_sql(
    sql="SELECT distinct(simulation) FROM market_meta", con=con
)["simulation"].values
st.session_state["available_sims"] = available_sims

sim = st.selectbox(label="Simulation", options=st.session_state["available_sims"])

##################
# market welfare #
##################
buyer_welfare = pd.read_sql(
    sql=f"""
        select
            sum(price - accepted_price) as buyer_welfare,
            start_time
        from market_orders
        where
            accepted_volume < 0
        and
            simulation = '{sim}'
        group by start_time
    """,
    con=con,
)
seller_welfare = pd.read_sql(
    sql=f"""
        select
            sum(accepted_price - price) as seller_welfare,
            start_time
        from market_orders
        where
            accepted_volume > 0
        and
            simulation = '{sim}'
        group by start_time
    """,
    con=con,
)
all_welfare = pd.merge(seller_welfare, buyer_welfare, on="start_time").sort_values(
    "start_time"
)
all_welfare["total_welfare"] = (
    all_welfare["buyer_welfare"] + all_welfare["seller_welfare"]
)
welfare_fig = px.line(
    data_frame=all_welfare,
    x="start_time",
    y=["seller_welfare", "buyer_welfare"],
    markers="*",
)
welfare_fig.update_layout(
    title="Market welfare",
    xaxis_title="Time",
    yaxis_title="Total welfare in ct.",
)
st.plotly_chart(welfare_fig)

lcol, mcol, rcol = st.columns(3)
lcol.write(f"Mean buyer welfare: {all_welfare['buyer_welfare'].mean():.2f}€")
mcol.write(f"Mean seller welfare: {all_welfare['seller_welfare'].mean():.2f}€")
rcol.write(f"Mean total welfare: {all_welfare['total_welfare'].mean():.2f}€")

##################
# market summary #
##################
meta_history = pd.read_sql(f"SELECT * FROM market_meta WHERE simulation = '{sim}'", con)
summary_fig = go.Figure()
summary_fig.add_trace(
    go.Scatter(
        x=meta_history["product_start"],
        y=meta_history["demand_volume"],
        yaxis="y",
        name="Demand volume",
    )
)
summary_fig.add_trace(
    go.Scatter(
        x=meta_history["product_start"],
        y=meta_history["supply_volume"],
        yaxis="y",
        name="Supply volume",
    )
)
summary_fig.add_trace(
    go.Scatter(
        x=meta_history["product_start"],
        y=meta_history["max_price"],
        yaxis="y2",
        name="Clearing price",
    )
)
summary_fig.update_layout(
    title="Market summary",
    yaxis={
        "side": "left",
        "range": [0, None],
    },
    yaxis2={"overlaying": "y", "side": "right"},
)
st.plotly_chart(summary_fig)
