from pathlib import Path

import pandas as pd
from itertools import product

from assume.common.utils import get_supported_solver
from mas4te.pricing_framework import PricingFramework, Storage

config_path = Path("example_data")

solar = pd.read_csv(
    config_path / "solar.csv", index_col=0, parse_dates=True
).squeeze()

demand = pd.read_csv(
    config_path / "demand.csv", index_col=0, parse_dates=True
).squeeze()

prices = pd.read_csv(
    config_path / "prices.csv", index_col=0, parse_dates=True
).squeeze()


import logging

logging.getLogger("gurobipy").setLevel("WARNING")

volumes = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35]

efficiencies = [0.5, 0.7, 0.9, 1]

objectives = []

from datetime import datetime
start = datetime(2023, 1, 1, hour=13)
end = datetime(2023, 1, 21, hour=13)

demand = demand[start:end]
prices = prices[start:end]
solar = solar[start:end]

# demand["demand"].plot()
# prices.plot()
# solar["solar"].plot()

# cost = demand["demand"] *prices["wholesale"]
# demand["demand"].sum()
# prices["wholesale"].mean()
# cost.sum()

dem_ids = list(range(30))

for volume, dem_id in product(volumes, dem_ids):
    print(volume, dem_id)
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=volume, efficiency=1),
        prices=prices.reset_index(drop=True),
        solar_generation=solar["solar"],
        demand=demand["demand_"+str(dem_id)],
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    
    objectives.append({
        "volume": volume,
        "dem_id": dem_id,
        "objective": pricer.model.objective(),
    })
    
print(objectives)

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

df = pd.DataFrame(objectives)
df = df.set_index("volume")

z_data = df.pivot(columns="dem_id")
x = z_data.columns.get_level_values(1)
fig = go.Figure(data=[go.Surface(z=z_data, y = list(z_data.index), x=x )])
fig.update_layout(title=dict(text='Storage sizes'), autosize=True,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90),
                scene=dict(
                        xaxis_title='Demand ID',
                        yaxis_title='Storage size in kWh',
                        zaxis_title='total cost',
                    ),
                  )
fig.show()

################ line
df = pd.DataFrame(objectives)

px.line(df, x="volume", y="objective", color="dem_id")