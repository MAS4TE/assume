# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st

st.set_page_config(layout="wide")

DB_URI = "postgresql://assume:assume@localhost:5432/assume"
con = psycopg2.connect(DB_URI)

sims = pd.read_sql("select distinct(simulation) FROM market_meta", con=con)[
    "simulation"
].values

n_parts_market_meta = {}
n_parts_orders = {}
for sim in sims:
    n_parts = pd.read_sql(
        f"select max(n_supply_units) from sim_config where simulation = '{sim}'",
        con=con,
    )["max"].values[0]

    market_meta = pd.read_sql(
        f"select * from market_meta where simulation = '{sim}' order by product_start asc",
        con,
    )
    n_parts_market_meta[n_parts] = market_meta

    market_orders = pd.read_sql(
        f"select * from market_orders where simulation = '{sim}' order by start_time asc",
        con,
    )
    n_parts_orders[n_parts] = market_orders


cp_fig = go.Figure()
for n_parts, market_meta in n_parts_market_meta.items():
    cp_fig.add_trace(
        go.Scatter(
            x=market_meta["product_start"],
            y=market_meta["max_price"] * 100,
            name=f"{n_parts}",
        )
    )
cp_fig.update_layout(
    title="Clearing price over time",
    legend=dict(title=dict(text="N participants")),
    yaxis_title="Price in ct./kWh",
)
st.plotly_chart(cp_fig)

traded_vol_fig = go.Figure()
for n_parts, market_meta in n_parts_market_meta.items():
    traded_vol_fig.add_trace(
        go.Scatter(
            x=market_meta["product_start"],
            y=market_meta["supply_volume"],
            name=f"{n_parts}",
        )
    )

    traded_vol_fig.add_trace(
        go.Scatter(
            x=market_meta["product_start"],
            y=market_meta["demand_volume"],
            name=f"{n_parts}",
        )
    )
traded_vol_fig.update_layout(
    title="Traded volume over time",
    legend=dict(title=dict(text="N participants")),
    yaxis_title="Volume in kWh",
)
st.plotly_chart(traded_vol_fig)

welfare_fig = go.Figure()
buyer_welfare_fig = go.Figure()
seller_welfare_fig = go.Figure()
for sim in sims:
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
    total_welfare = pd.merge(
        left=seller_welfare, right=buyer_welfare, on="start_time", how="inner"
    )
    total_welfare["total_welfare"] = (
        total_welfare["buyer_welfare"] + total_welfare["seller_welfare"]
    )
    total_welfare.sort_values("start_time", inplace=True)

    n_parts = pd.read_sql(
        f"select max(n_supply_units) from sim_config where simulation = '{sim}'",
        con=con,
    )["max"].values[0]

    welfare_fig.add_trace(
        go.Scatter(
            x=total_welfare["start_time"],
            y=total_welfare["total_welfare"],
            name=f"{n_parts}",
        )
    )
    buyer_welfare_fig.add_trace(
        go.Scatter(
            x=total_welfare["start_time"],
            y=total_welfare["buyer_welfare"],
            name=f"{n_parts}",
        )
    )
    seller_welfare_fig.add_trace(
        go.Scatter(
            x=total_welfare["start_time"],
            y=total_welfare["seller_welfare"],
            name=f"{n_parts}",
        )
    )
welfare_fig.update_layout(
    title="Welfare over time",
    legend=dict(title=dict(text="N participants")),
    yaxis_title="Welfare in €",
)
buyer_welfare_fig.update_layout(
    title="Buyer welfare over time",
    legend=dict(title=dict(text="N participants")),
    yaxis_title="Welfare in €",
)
seller_welfare_fig.update_layout(
    title="Seller welfare over time",
    legend=dict(title=dict(text="N participants")),
    yaxis_title="Welfare in €",
)
st.plotly_chart(welfare_fig)
st.plotly_chart(buyer_welfare_fig)
st.plotly_chart(seller_welfare_fig)
