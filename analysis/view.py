# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import streamlit as st

st.set_page_config(layout="wide")

st.title("Analysis pages for storage market simulations")

st.header("Following simulations have been done:")
st.write(
    "Baseline scenario - 50 supply & 50 demand units. Supply units all have 5kWh storage. 30 runs with different profiles"
)
st.write(
    "Supply surplus scenario - 50 supply & 50 demand units. Supply units all have 10kWh storage. 30 runs with different profiles"
)
st.write(
    "Demand surplus scenario - 50 supply & 50 demand units. Supply units all have 2.5kWh storage. 30 runs with different profiles"
)

st.header("Following pages are available:")
st.page_link(page="./pages/clearing_prices.py", label="Clearing prices")
st.page_link(page="./pages/welfare.py", label="Market welfare")
