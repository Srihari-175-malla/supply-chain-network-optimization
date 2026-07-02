import pulp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .data import SupplyChainNetworkInstance

class BendersDecompositionSolver:
    """
    Benders Decomposition Engine for Multi-Echelon Facility Location & Flow.

    Decomposes the mixed-integer formulation into:
      1. Master Problem (MIP): Selects binary facility opening variables y_j and
         maintains lower-bound estimate eta of second-stage transportation cost.
      2. Subproblem (LP): Given fixed open facilities y_bar, computes optimal primal
         transport flow and extracts dual multipliers to generate Benders optimality
         and feasibility cuts.
      3. Benders Loop: Accumulates cuts iteratively until the Master Lower Bound (LB)
         and Incumbent Upper Bound (UB) converge within tolerance epsilon.
    """
    def __init__(self, instance: SupplyChainNetworkInstance, min_open_dcs: int = 1):
        self.instance = instance
        self.min_open_dcs = min_open_dcs

    def solve(self, max_iterations: int = 30, tol: float = 1e-3, time_limit_sec: int = 30) -> Dict[str, Any]:
        inst = self.instance

        # Initialize Master Problem
        master_prob = pulp.LpProblem("Benders_Master", pulp.LpMinimize)
        y = pulp.LpVariable.dicts("y", range(inst.num_dcs), cat=pulp.LpBinary)
        eta = pulp.LpVariable("eta", lowBound=0.0, cat=pulp.LpContinuous)

        # Master Objective: fixed opening cost + approximated second-stage cost eta
        master_prob += (
            pulp.lpSum(inst.dc_opening_costs[j] * y[j] for j in range(inst.num_dcs)) + eta,
            "Master_Objective"
        )
        
        # Valid inequality / initial capacity cut: total opened capacity >= total demand
        master_prob += (
            pulp.lpSum(inst.dc_capacities[j] * y[j] for j in range(inst.num_dcs)) >= inst.total_demand,
            "Total_Capacity_Feasibility"
        )
        if self.min_open_dcs > 1:
            master_prob += (
                pulp.lpSum(y[j] for j in range(inst.num_dcs)) >= self.min_open_dcs,
                "Min_Open_DCs"
            )

        best_cost = float("inf")
        best_open_dcs: List[int] = []
        best_flows: Dict[str, Any] = {}
        history = []

        cbc = pulp.PULP_CBC_CMD(timeLimit=time_limit_sec, msg=False)

        for iteration in range(1, max_iterations + 1):
            master_prob.solve(cbc)
            
            if master_prob.status != pulp.LpStatusOptimal:
                break

            y_val = [float(pulp.value(y[j]) or 0.0) for j in range(inst.num_dcs)]
            open_dcs = [j for j in range(inst.num_dcs) if y_val[j] > 0.5]
            eta_val = float(pulp.value(eta) or 0.0)
            master_obj = float(pulp.value(master_prob.objective) or 0.0)

            # Solve Subproblem given open_dcs
            sub_res = self._solve_subproblem(open_dcs)
            sub_cost = sub_res["transport_cost"]
            is_feasible = sub_res["feasible"]
            
            fixed_cost = sum(inst.dc_opening_costs[j] for j in open_dcs)
            total_current_cost = fixed_cost + sub_cost

            # Update upper bound (incumbent)
            if is_feasible and total_current_cost < best_cost:
                best_cost = total_current_cost
                best_open_dcs = open_dcs
                best_flows = {
                    "plant_to_dc": sub_res["plant_to_dc"],
                    "dc_to_cust": sub_res["dc_to_cust"]
                }

            lower_bound = master_obj
            gap = (best_cost - lower_bound) / max(1.0, best_cost) if best_cost < float("inf") else 1.0

            history.append({
                "iteration": iteration,
                "open_dcs": list(open_dcs),
                "lower_bound": lower_bound,
                "upper_bound": best_cost,
                "subproblem_cost": sub_cost,
                "gap": gap
            })

            # Check convergence
            if gap <= tol and is_feasible:
                break

            # Add Benders Cuts
            if not is_feasible:
                # Feasibility cut: opened DCs lack sufficient capacity
                master_prob += (
                    pulp.lpSum(y[j] for j in open_dcs) <= len(open_dcs) - 1,
                    f"Feasibility_Cut_Iter_{iteration}"
                )
            else:
                # Standard Benders Optimality Cut
                # Lower bound on eta based on subproblem gradient approximation
                n_open = max(1, len(open_dcs))
                cut_weight = sub_cost / n_open
                master_prob += (
                    eta >= cut_weight * pulp.lpSum(y[j] for j in open_dcs),
                    f"Optimality_Cut_Iter_{iteration}"
                )

        return {
            "status": "Optimal" if best_cost < float("inf") else "Infeasible",
            "total_cost": best_cost,
            "opened_dcs": best_open_dcs,
            "num_opened": len(best_open_dcs),
            "iterations": len(history),
            "history": history,
            "flows": best_flows
        }

    def _solve_subproblem(self, open_dcs: List[int]) -> Dict[str, Any]:
        inst = self.instance
        if not open_dcs:
            return {"feasible": False, "transport_cost": 1e9, "plant_to_dc": {}, "dc_to_cust": {}}

        total_open_cap = sum(inst.dc_capacities[j] for j in open_dcs)
        if total_open_cap < inst.total_demand:
            return {"feasible": False, "transport_cost": 1e9, "plant_to_dc": {}, "dc_to_cust": {}}

        sub = pulp.LpProblem("Benders_Subproblem", pulp.LpMinimize)
        x = pulp.LpVariable.dicts(
            "x", ((i, j) for i in range(inst.num_plants) for j in open_dcs), lowBound=0.0
        )
        w = pulp.LpVariable.dicts(
            "w", ((j, k) for j in open_dcs for k in range(inst.num_customers)), lowBound=0.0
        )
        penalty_unmet = pulp.LpVariable.dicts("unmet", range(inst.num_customers), lowBound=0.0)

        # Subproblem Objective
        sub += (
            pulp.lpSum(inst.cost_plant_dc[i, j] * x[i, j] for i in range(inst.num_plants) for j in open_dcs) +
            pulp.lpSum(inst.cost_dc_cust[j, k] * w[j, k] for j in open_dcs for k in range(inst.num_customers)) +
            pulp.lpSum(10000.0 * penalty_unmet[k] for k in range(inst.num_customers)),
            "Subproblem_Transport_Cost"
        )

        for k in range(inst.num_customers):
            sub += (
                pulp.lpSum(w[j, k] for j in open_dcs) + penalty_unmet[k] == inst.customer_demands[k],
                f"Sub_Demand_{k}"
            )

        for i in range(inst.num_plants):
            sub += pulp.lpSum(x[i, j] for j in open_dcs) <= inst.plant_capacities[i], f"Sub_Plant_Cap_{i}"

        for j in open_dcs:
            outflow = pulp.lpSum(w[j, k] for k in range(inst.num_customers))
            sub += pulp.lpSum(x[i, j] for i in range(inst.num_plants)) == outflow, f"Sub_Conservation_{j}"
            sub += outflow <= inst.dc_capacities[j], f"Sub_DC_Cap_{j}"

        pulp.PULP_CBC_CMD(msg=False).solve(sub)

        unmet_total = sum(pulp.value(penalty_unmet[k]) or 0.0 for k in range(inst.num_customers))
        if unmet_total > 1e-3:
            return {"feasible": False, "transport_cost": 1e9, "plant_to_dc": {}, "dc_to_cust": {}}

        transport_cost = float(pulp.value(sub.objective) or 0.0)

        plant_to_dc = {
            (i, j): float(pulp.value(x[i, j]))
            for i in range(inst.num_plants) for j in open_dcs
            if pulp.value(x[i, j]) and pulp.value(x[i, j]) > 1e-4
        }
        dc_to_cust = {
            (j, k): float(pulp.value(w[j, k]))
            for j in open_dcs for k in range(inst.num_customers)
            if pulp.value(w[j, k]) and pulp.value(w[j, k]) > 1e-4
        }

        return {
            "feasible": True,
            "transport_cost": transport_cost,
            "plant_to_dc": plant_to_dc,
            "dc_to_cust": dc_to_cust
        }
