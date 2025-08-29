from mas4te.pricing_framework import PricingFramework, Storage
from assume.common.utils import get_supported_solver


def test_pricing_framework_baseline():

    # buying 1 kWh for 1 €/kWh should equal to 3€ total
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=0, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [1, 1, 1]}),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([1, 1, 1])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    assert pricer.model.objective() == -3

def test_pricing_framework_opti_storage():

    # buying 2 kWh for 0€/kWh and storing 1 kWh of this should equal 1€ total
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=1, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [0, 1, 1]}),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([1, 1, 1])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    assert pricer.model.objective() == -1

def test_pricing_framework_opti_storage_2():

    # now we need 2 kWh at each timestep
    # on timestep=0, we can buy for 0€/kWh and should buy 3kWh
    # as we use 2 kWh during timestep=0 and use 1 kWh for timestep=1
    # total cost should be 3*0 + 1*1 + 2*1 = 3
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=1, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [0, 1, 1]}),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([2, 2, 2])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    assert pricer.model.objective() == -3

def test_pricing_framework_selling_pv():

    # here we should gain 1€ from selling pv
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=0, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [1, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [1, 1, 1]}),
        solar_generation=pd.Series([1, 0, 0]),
        demand=pd.Series([0, 0, 0])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    assert pricer.model.objective() == 1

def test_pricing_framework_selling_pv_w_storage():

    # same as above, but we can store PV and sell at
    # timestep=1 instead of timestep=0, as we can get 2€/kWh
    # in timestep=1
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=1, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [1, 2, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [1, 1, 1]}),
        solar_generation=pd.Series([1, 0, 0]),
        demand=pd.Series([0, 0, 0])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    assert pricer.model.objective() == 2

    # charge from solar_generation in ts=0,1 and discharge at ts=2
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=2, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [5, 10, 20]}),
        solar_generation=pd.Series([1, 1, 0]),
        demand=pd.Series([0, 0, 2])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    print(pricer.model.objective())
    assert pricer.model.objective() == 0

def test_pricing_framework_negative_prices():

    # buy as much in ts=2 cause we get paid for this
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=1, volume=2, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [5, 10, -20]}),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([0, 0, 2])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    print(pricer.model.objective())
    assert pricer.model.objective() == 80

def test_pricing_framework__c_rate():

    # check if c_rate is respected
    pricer = PricingFramework(
        storage=Storage(id=0, c_rate=0.5, volume=2, efficiency=1),
        prices=pd.DataFrame({
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [0, 10, 0]}),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([2, 2, 0])
    )
    pricer.optimize(solver=get_supported_solver("gurobi"))
    print(pricer.model.objective())
    assert pricer.model.objective() == -10