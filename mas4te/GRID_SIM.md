<!--
SPDX-FileCopyrightText: MAS4TE Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Grid simulation: ASSUME with BUC bidding

This guide is for anyone who builds a grid simulation (or any other analysis) on the **bids, awards and schedules** of the MAS4TE battery market. You provide your own load profiles (or synload-de profiles), ASSUME simulates the weekly market, every household bids with the Battery Utility Calculator (BUC), and the results end up as CSV files.

It needs **no** MQTT broker, no communications agents, no LLM, no Chronos service and no blockchain setup. Everything runs in one Python process.

Branch: `grid-sim` in the `assume` repo (folder `mas4te/`).

## 1. What a run does

```
inputs/ (loads, PV, prices, households)
   │
   ▼  for every week (the market opens weekly)
 every household:  BUC values storage sizes  →  bidding curve  →  one bid per kWh step
 market (BatteryClearing, HiGHS):  clears the bids, uniform price per location, counterparties (trade_id)
 every awarded household:  BUC computes the schedule for the awarded volume
   │
   ▼
results/<run>/  bids.csv  trades.csv  household_flows.csv  household_summary.csv  battery_schedule.csv
```

- **Buyers** (`buy`) have no storage and rent virtual storage capacity from sellers. The weekly product is the rent for 7 days, priced in EUR per kWh of storage and week.
- **Sellers** (`sell`) own a physical battery and offer a part of it (`tradeable_pct`). They use the rest themselves.
- The market opens weekly at the start time (`--start`); a product is delivered for one week, starting 12 hours after the opening.
- When bidding, the BUC sees the **actual profiles of the delivery week** (perfect foresight, see section 6).

## 2. Installation

The environment is the ASSUME one. The folder `assume` has a ready `.venv`, otherwise:

```bash
cd assume
python -m venv .venv && source .venv/bin/activate
pip install -e .          # brings pyomo, highspy and the BUC
pip install pyarrow       # only for Parquet files (synload-de, --format parquet)
```

The solver is `appsi_highs`, no license needed. Use `--solver gurobi` for Gurobi.

All commands below run in `assume/mas4te`.

## 3. Quick start

```bash
cd assume/mas4te

# small synthetic example (2 buyers, 1 seller, 6 weeks of data)
python make_inputs.py synthetic --out inputs/example --buyers 2 --sellers 1

# simulate 2 weeks
python simulation.py --inputs inputs/example --out results/example --weeks 2
```

This takes about a minute. Afterwards the files from section 5 are in `results/example/`.

The synthetic example only exists to try things out (sine curves, fixed prices). For real results you need your own data.

## 4. Input data

One folder with four files. CSV or Parquet (the file extension decides), `solar` is optional.

### `units.csv`: one household per row

| Column | Required | Meaning |
|---|---|---|
| `unit_id` | yes | unique name, e.g. `B_01` (buyer), `S_01` (seller) |
| `role` | yes | `buy` or `sell` |
| `load_profile` | yes | column name in `loads` |
| `solar_profile` | no | column name in `solar`, empty = no PV |
| `location` | no | location, default `aachen`. Allowed are the locations of the BUC grid-fee table (`aachen`, `juelich`, `heerlen`, `liege`) |
| `battery_kwh` | sellers | battery capacity in kWh, 0 for buyers |
| `tradeable_pct` | no | share of the battery that may be rented out, default 50 |
| `risk_tolerance` | no | `Low` (0.05), `Medium` (0.35), `High` (0.50), default `Low`; only used with `--risk` |
| `trading_scope` | no | `All` (rent at all locations) or `Community` (own location only), default `All` |

### `loads.csv` and `solar.csv`: time series in kW

A column `datetime` and one column per profile. Power in **kW** (mean over the time step). The time step is detected from the loads (e.g. 15 minutes). Time zones are converted to UTC and dropped, so convert to the zone you want to analyse in yourself and give timestamps without a zone if you want local time.

```csv
datetime,load_0,load_1
2023-05-01 00:00:00,0.41,0.88
2023-05-01 00:15:00,0.39,0.91
```

The data must cover the **whole** simulation period plus the delivery week of the last auction (up to `start + weeks·7 days + 12 h`). Otherwise the run stops with a message saying how far the data has to reach.

**Own load profiles:** put them as columns into `loads.csv` and refer to them in `units.csv`.

**synload-de profiles:** `make_inputs.py synload` adds the four components (`base_load_kw`, `hp_load_kw`, `dhw_load_kw`, `ev_load_kw`) to one load per profile:

```bash
python make_inputs.py synload --out inputs/synload \
    --timeseries ~/code/synload-de/output/timeseries.parquet \
    --prices ~/code/battery-utility-calculator/data/prices.csv \
    --pv ~/code/battery-utility-calculator/data/pv.parquet \
    --profiles 0 1 2 3 --sellers 1
```

`--sellers 1` turns the last of the given profiles into a seller (with a 10 kWh battery). Adjust `units.csv` by hand afterwards (battery sizes, locations, risk). The PV file is only taken roughly (first numeric column), check `solar.csv` to see whether it is the plant you want. synload-de has no PV, it comes from the BUC repo or from you.

### `prices.csv`: prices in EUR/kWh

Columns `datetime`, `wholesale`, `eeg`, `grid` (household electricity price, `supplier` is accepted too) and `community`. One price per column for all households, the community price applies to all locations.

## 5. Results

All files go to `--out`. Times are those of the input data. With `--format parquet` the files are `.parquet`. The ASSUME database `assume.db` is next to them (mostly uninteresting for you).

| File | Content |
|---|---|
| `bids.csv` | **Every** bid with its result: `unit_id`, `role`, `bid_id`, `product_start/end`, `location`, `exclusive_id`, `bid_volume_kwh`, `bid_price_eur_kwh`, `accepted_volume_kwh`, `accepted_price_eur_kwh`. One bid is one kWh step of the bidding curve. |
| `trades.csv` | Every award as a pair: `trade_id`, `buyer_unit`, `seller_unit`, `volume_kwh`, `price_eur_kwh`, `location`, bid ids. |
| `household_flows.csv` | Time series per **awarded** household and week, in the resolution of the input data, columns see below. |
| `household_summary.csv` | One row per household and week: `awarded_kwh`, `own_storage_kwh`, `rent_eur` (buyers negative, sellers positive), `cashflow_*_eur` (BUC cost components: community, supplier, eeg, wholesale, grid_fees; revenues positive, costs negative). |
| `battery_schedule.csv` | **Physical battery of the sellers**: `own_use_kw`, `rented_out_kw`, `total_kw`. |

Only households that were **awarded** appear (buyers with an award, sellers that rented out something). With `--export-unawarded` all others are written too (buyers without storage, sellers with full own use).

### Columns of `household_flows.csv`

All power values in **kW** (mean per time step), `kwh` only for storage levels.

| Column | Meaning |
|---|---|
| `timestamp`, `unit_id`, `role`, `location`, `product_start` | time, household, week |
| `demand_kw`, `solar_kw` | input profiles |
| `grid_import_kw` | draw at the grid connection: supplier, wholesale and community, directly or into the storage |
| `grid_export_kw` | feed-in at the grid connection: PV or storage to EEG, wholesale or community |
| `net_grid_kw` | `grid_import_kw − grid_export_kw`, positive = draw |
| `net_grid_without_storage_kw` | `demand − solar`, the load without storage trading (for comparison) |
| `storage_soc_kwh`, `soc_<eeg/wholesale/community/home>_kwh` | level of the (virtual) storage in total and per use case |
| `storage_charge_kw` | change of the level per hour, positive = charging, negative = discharging (before conversion losses) |
| `flow_<source>_to_<sink>_kw` | single BUC flows, if you want to recombine them (e.g. `flow_pv_to_home_kw`, `flow_supplier_to_storage_kw`, `flow_community_to_home_kw`, `flow_storage_to_home_kw`) |

**Important for the grid simulation:**

- For a **buyer** the flows describe a *virtual* storage as if it stood at the buyer's grid connection. `net_grid_kw` is what the buyer's meter sees when the rented capacity is used. The battery that actually charges and discharges is at the seller.
- The **physical battery of a seller** is in `battery_schedule.csv`: `own_use_kw` (the seller's own part, from the seller's schedule) plus `rented_out_kw` (the schedules of the tenants, split by the matched volumes, see `trades.csv`). `total_kw` is the power the battery really runs. The load flow at the seller's connection (`net_grid_kw` in `household_flows.csv`) contains only the own part. Depending on your grid model you still have to add the rented part: `total_kw − own_use_kw`.
- Trading flows (community, wholesale, EEG) are already counted in `grid_import_kw`/`grid_export_kw` as physical draw and feed-in. How community energy actually flows through the grid (between which households) is not provided by this simulation, that is the job of the grid simulation.

## 6. Assumptions and limits

- **Perfect foresight:** the BUC bids on the actual profiles of the delivery week. The communications agent uses forecasts (Chronos). Forecast errors are not built in yet, so the bids are better informed than in reality.
- **Deterministic bidding is the default.** With `--risk` the BUC computes `--scenarios` disturbed scenarios (seed `--seed`) and bids a quantile: buyers take the lower, sellers the upper end, so asks and bids move apart and **less trade** happens than without `--risk`. `risk_tolerance` per household controls how far.
- **Buyer volume:** the agent asks an LLM for the upper bound of the search; here it is `max(5, min(10, peak net load + 3))` kWh. The 10 is `BUCSettings.max_buy_volume_kwh`.
- **C-rate:** the storages in the BUC have C-rate 0.2 (as in the agent); the market only allows C-rate 1 in the bid field. The agent does the same.
- **Time step:** the BUC calculates with the time step of the input data (`hours_per_timestep`). The communications agent currently does not pass this value (BUC default 1 hour) while feeding the BUC 15-minute data, which is probably a bug in the agent. The bids here are therefore **not identical** to those of the agent.
- **Several locations:** buyers bid at every location in their `trading_scope`, the market awards exclusively (`exclusive_id`). The schedule of a buyer that is awarded at several locations is computed for the location with the largest share (warning in the log). The agent's country scope (`Country`) does not exist here.
- **EV:** the EV load from synload-de is part of the fixed household load and is not optimised. The flexible BUC `ElectricVehicle` is not connected to this simulation (yet); the helper functions for it remain unchanged in `simulation.py`.
- **Blockchain/BRP:** no results are sent to the BRP (`BRP_MARKET_RESULTS_URL` is removed), and the blockchain addresses in the bids are placeholders.
- **Sellers without an award** are not in the schedules unless `--export-unawarded` is set. Their battery is then not listed in `battery_schedule.csv`.
- **No grid model:** voltages, lines and grid fees between households are not part of this simulation. Grid fees only enter the valuation through the location table in the BUC.

## 7. Options of `simulation.py`

| Option | Default | Meaning |
|---|---|---|
| `--inputs` | – | input folder |
| `--out` | – | result folder |
| `--weeks` | 4 | number of weekly markets (data up to `start + weeks·7 days + 12 h` needed) |
| `--start` | start of the data | first market opening, e.g. `2023-05-08T13:00` |
| `--risk` | off | risk adjusted bidding curve instead of the deterministic one |
| `--scenarios` | 12 | scenarios for risk adjusted bidding |
| `--seed` | 42 | seed of the scenarios, `-1` = different every time |
| `--goal` | `max_cashflow` | or `max_green_energy` (this also turns `--risk` off) |
| `--solver` | `appsi_highs` | Pyomo solver of the BUC |
| `--format` | `csv` | or `parquet` |
| `--export-unawarded` | off | also write households without an award |
| `--log-level` | `WARNING` | `INFO` shows the bids per household and week |

## 8. Runtime and typical messages

- The time is spent in the BUC: for every household and week one optimisation problem is solved per storage size, with `--risk` additionally per scenario. Rough order of magnitude on a laptop: 3 households, 2 weeks, deterministic about 1 minute; with `--risk` and 3 scenarios 3 households and 1 week also take about 1 minute. For many households, test with few weeks first.
- `ERROR ... no such table: market_dispatch` / `rl_params` at the end comes from ASSUME's KPI query when nothing was traded and is harmless.
- `no household schedules recorded, nothing was awarded`: nothing was awarded. Usually the sellers' prices are above the buyers' prices (typical with `--risk`); check `bids.csv`.
- `input data covers X of Y steps`: the input data does not reach the last delivery week; reduce `--weeks` or extend the data.

## 9. Where is what

| File | Purpose |
|---|---|
| `simulation.py` | builds the world (market, households), command line |
| `buc_strategy.py` | `BUCStrategy`: bidding curve via the BUC, schedule after the clearing |
| `grid_sim_io.py` | reads the inputs, `Recorder` for the results |
| `make_inputs.py` | creates an input folder (synthetic or from synload-de) |
| `battery_market_clearing.py` | market clearing (unchanged) |
| `mas4te_bidding_strategy.py` | bidding over MQTT with the agents (unchanged, not used here) |
| `../tests/test_mas4te_grid_sim.py` | tests for inputs and results, no solver needed |
