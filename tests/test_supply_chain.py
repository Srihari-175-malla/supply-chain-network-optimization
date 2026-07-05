import pytest
import numpy as np
from src.supply_chain_opt.data import (
    SupplyChainNetworkInstance,
    generate_synthetic_supply_chain_network,
    create_cflp_benchmark_instance
)
from src.supply_chain_opt.milp import MultiEchelonCFLPSolver
from src.supply_chain_opt.benders import BendersDecompositionSolver
from src.supply_chain_opt.stochastic import TwoStageStochasticCFLPSolver
from src.supply_chain_opt.robust import RobustCFLPSolver
from src.supply_chain_opt.evaluation import OutOfSampleEvaluator

def test_data_model():
    inst = generate_synthetic_supply_chain_network(n_plants=3, n_dcs=6, n_cust=15, seed=42)
    assert inst.num_plants == 3
    assert inst.num_dcs == 6
    assert inst.num_customers == 15
    assert inst.total_demand > 0
    assert inst.total_plant_capacity >= inst.total_demand
    assert inst.total_dc_capacity >= inst.total_demand
    assert inst.cost_plant_dc.shape == (3, 6)
    assert inst.cost_dc_cust.shape == (6, 15)

def test_benchmark_instance_loader():
    inst = create_cflp_benchmark_instance("cap71")
    assert inst.num_plants == 4
    assert inst.num_dcs == 8
    assert inst.num_customers == 30

def test_milp_solver_continuous():
    inst = generate_synthetic_supply_chain_network(n_plants=2, n_dcs=4, n_cust=10, seed=123)
    solver = MultiEchelonCFLPSolver(inst, single_sourcing=False)
    res = solver.solve()

    assert res["status"] == "Optimal"
    assert res["total_cost"] > 0
    assert res["num_opened"] >= 1
    assert len(res["opened_dcs"]) == res["num_opened"]
    assert len(res["plant_to_dc"]) > 0
    assert len(res["dc_to_cust"]) > 0

    # Total flow across DCs to customers must match total demand
    total_cust_flow = sum(res["dc_to_cust"].values())
    assert pytest.approx(total_cust_flow, rel=1e-3) == inst.total_demand

def test_milp_solver_single_sourcing():
    inst = generate_synthetic_supply_chain_network(n_plants=2, n_dcs=4, n_cust=8, seed=456)
    solver = MultiEchelonCFLPSolver(inst, single_sourcing=True)
    res = solver.solve()

    assert res["status"] == "Optimal"
    assert res["num_opened"] >= 1
    # Each customer should be served by exactly 1 DC
    served_customers = [k for (_, k) in res["dc_to_cust"].keys()]
    assert len(served_customers) == inst.num_customers

def test_benders_decomposition():
    inst = generate_synthetic_supply_chain_network(n_plants=2, n_dcs=4, n_cust=10, seed=789)
    solver = BendersDecompositionSolver(inst, min_open_dcs=1)
    res = solver.solve(max_iterations=15)

    assert res["status"] == "Optimal"
    assert res["total_cost"] > 0
    assert res["num_opened"] >= 1
    assert res["iterations"] >= 1

def test_two_stage_stochastic():
    inst = generate_synthetic_supply_chain_network(n_plants=2, n_dcs=4, n_cust=10, seed=321)
    solver = TwoStageStochasticCFLPSolver(inst, num_scenarios=5, demand_volatility=0.20, seed=42)
    res = solver.solve()

    assert res["status"] == "Optimal"
    assert res["expected_cost"] > 0
    assert res["num_opened"] >= 1
    assert len(res["scenario_costs"]) == 5

def test_robust_optimization():
    inst = generate_synthetic_supply_chain_network(n_plants=2, n_dcs=4, n_cust=10, seed=654)
    solver = RobustCFLPSolver(inst, uncertainty_factor=0.20, gamma=3.0)
    res = solver.solve()

    assert res["status"] == "Optimal"
    assert res["robust_cost"] > 0
    assert res["num_opened"] >= 1

def test_out_of_sample_evaluator():
    inst = generate_synthetic_supply_chain_network(n_plants=2, n_dcs=4, n_cust=10, seed=987)
    evaluator = OutOfSampleEvaluator(inst, num_oos_scenarios=10, seed=42)
    res = evaluator.evaluate_facility_placement([0, 1, 2])

    assert res["mean_cost"] > 0
    assert res["worst_cost"] >= res["mean_cost"]
    assert len(res["scenario_costs"]) == 10
