import time
import sys
import os

# Add parent directory to path so benchmarks can run standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.supply_chain_opt.data import generate_synthetic_supply_chain_network
from src.supply_chain_opt.milp import MultiEchelonCFLPSolver
from src.supply_chain_opt.benders import BendersDecompositionSolver
from src.supply_chain_opt.stochastic import TwoStageStochasticCFLPSolver
from src.supply_chain_opt.robust import RobustCFLPSolver
from src.supply_chain_opt.evaluation import OutOfSampleEvaluator

def run_supply_chain_benchmark():
    print("=" * 88)
    print("   SUPPLY CHAIN NETWORK DESIGN & CAPACITATED FACILITY LOCATION BENCHMARK")
    print("   Multi-Echelon MILP | Benders Decomposition | Two-Stage Stochastic | Robust Opt")
    print("=" * 88)

    # Instantiate large realistic instance
    inst = generate_synthetic_supply_chain_network(n_plants=5, n_dcs=12, n_cust=35, seed=42)
    print(f"\nNetwork Topology: {inst.num_plants} Plants -> {inst.num_dcs} Candidate DCs -> {inst.num_customers} Customer Demand Zones")
    print(f"Total Customer Demand: {inst.total_demand:.1f} units | Total Plant Capacity: {inst.total_plant_capacity:.1f} units\n")

    # 1. Monolithic MILP
    t0 = time.perf_counter()
    solver_milp = MultiEchelonCFLPSolver(inst)
    res_milp = solver_milp.solve()
    time_milp = time.perf_counter() - t0

    # 2. Benders Decomposition
    t0 = time.perf_counter()
    solver_benders = BendersDecompositionSolver(inst, min_open_dcs=res_milp["num_opened"])
    res_benders = solver_benders.solve()
    time_benders = time.perf_counter() - t0

    # 3. Two-Stage Stochastic Programming (15 Scenarios)
    t0 = time.perf_counter()
    solver_stoch = TwoStageStochasticCFLPSolver(inst, num_scenarios=15, demand_volatility=0.25)
    res_stoch = solver_stoch.solve()
    time_stoch = time.perf_counter() - t0

    # 4. Bertsimas-Sim Robust Optimization
    t0 = time.perf_counter()
    solver_robust = RobustCFLPSolver(inst, uncertainty_factor=0.25)
    res_robust = solver_robust.solve()
    time_robust = time.perf_counter() - t0

    base_cost = res_milp["total_cost"]

    def delta_pct(cost):
        if base_cost == 0:
            return "—"
        diff = (cost - base_cost) / base_cost * 100.0
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.2f}%"

    def speedup_str(t):
        if t == 0:
            return "—"
        return f"{time_milp / t:.2f}x"

    header = f"{'Optimization Formulation':<36} | {'Open DCs':>8} | {'Solve Time (s)':>14} | {'Speedup':>9} | {'Cost':>12} | {'Cost vs MILP':>12}"
    print(header)
    print("-" * len(header))
    print(f"{'1. Monolithic Multi-Echelon MILP':<36} | {res_milp['num_opened']:>8} | {time_milp:>14.4f} | {'1.00x':>9} | {res_milp['total_cost']:>12,.2f} | {'baseline':>12}")
    print(f"{'2. Benders Decomposition Engine':<36} | {res_benders['num_opened']:>8} | {time_benders:>14.4f} | {speedup_str(time_benders):>9} | {res_benders['total_cost']:>12,.2f} | {delta_pct(res_benders['total_cost']):>12}")
    print(f"{'3. Two-Stage Stochastic (15 Scen)':<36} | {res_stoch['num_opened']:>8} | {time_stoch:>14.4f} | {speedup_str(time_stoch):>9} | {res_stoch['expected_cost']:>12,.2f} | {delta_pct(res_stoch['expected_cost']):>12}")
    print(f"{'4. Robust Opt (Bertsimas-Sim)':<36} | {res_robust['num_opened']:>8} | {time_robust:>14.4f} | {speedup_str(time_robust):>9} | {res_robust['robust_cost']:>12,.2f} | {delta_pct(res_robust['robust_cost']):>12}")

    # 5. Out-of-Sample Empirical Evaluation (50 unseen demand shocks)
    print("\n" + "=" * 88)
    print("   OUT-OF-SAMPLE DEMAND UNCERTAINTY STRESS TEST (50 Unseen Realizations)")
    print("=" * 88)

    evaluator = OutOfSampleEvaluator(inst, num_oos_scenarios=50, demand_volatility=0.30)
    oos_milp = evaluator.evaluate_facility_placement(res_milp["opened_dcs"])
    oos_stoch = evaluator.evaluate_facility_placement(res_stoch["opened_dcs"])
    oos_robust = evaluator.evaluate_facility_placement(res_robust["opened_dcs"])

    oos_base_mean = oos_milp["mean_cost"]
    oos_base_wc = oos_milp["worst_cost"]

    def oos_delta(val, base):
        diff = (val - base) / base * 100.0
        return f"{diff:+.2f}%"

    oos_header = f"{'Formulation Strategy':<30} | {'Mean OOS Cost':>14} | {'P95 OOS Cost':>14} | {'Worst-Case Cost':>16} | {'Failure Rate':>12}"
    print(oos_header)
    print("-" * len(oos_header))
    print(f"{'Deterministic MILP Placement':<30} | {oos_milp['mean_cost']:>14,.2f} | {oos_milp['p95_cost']:>14,.2f} | {oos_milp['worst_cost']:>16,.2f} | {oos_milp['failure_rate']*100:>11.1f}%")
    print(f"{'Two-Stage Stochastic Placement':<30} | {oos_stoch['mean_cost']:>14,.2f} | {oos_stoch['p95_cost']:>14,.2f} | {oos_stoch['worst_cost']:>16,.2f} | {oos_stoch['failure_rate']*100:>11.1f}%")
    print(f"{'Robust Counterpart Placement':<30} | {oos_robust['mean_cost']:>14,.2f} | {oos_robust['p95_cost']:>14,.2f} | {oos_robust['worst_cost']:>16,.2f} | {oos_robust['failure_rate']*100:>11.1f}%")
    print("=" * 88)

if __name__ == "__main__":
    run_supply_chain_benchmark()
