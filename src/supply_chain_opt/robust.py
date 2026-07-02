import pulp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .data import SupplyChainNetworkInstance

class RobustCFLPSolver:
    """
    Bertsimas-Sim (Gamma-Budget) Robust Optimization Solver for Facility Location.

    Protects against bounded interval demand uncertainty:
      d_k in [d_bar_k, d_bar_k + d_hat_k],  where d_hat_k = uncertainty_factor * d_bar_k
    
    The budget of uncertainty Gamma in [0, |K|] bounds the number of customer demands
    that can simultaneously surge to their peak worst-case value.

    The robust constraint ensures that even under the worst-case scenario within the
    uncertainty set, total opened DC capacity and flow meet customer demand:
      sum_k d_bar_k * w_jk + max_{S subseteq K, |S| <= Gamma} ( sum_{k in S} d_hat_k * w_jk ) <= Cap_j * y_j
    
    Using LP duality, the inner maximization is reformulated into linear constraints
    with auxiliary dual variables z_j and p_jk.
    """
    def __init__(
        self,
        instance: SupplyChainNetworkInstance,
        uncertainty_factor: float = 0.25,
        gamma: Optional[float] = None
    ):
        self.instance = instance
        self.uncertainty_factor = uncertainty_factor
        # Default budget: protect against ~30% of customers surging simultaneously
        self.gamma = gamma if gamma is not None else float(max(1.0, np.round(0.3 * instance.num_customers)))

    def solve(self, time_limit_sec: int = 30) -> Dict[str, Any]:
        inst = self.instance
        prob = pulp.LpProblem("Robust_CFLP_BertsimasSim", pulp.LpMinimize)

        # Decision Variables
        y = pulp.LpVariable.dicts("open_dc", range(inst.num_dcs), cat=pulp.LpBinary)
        x = pulp.LpVariable.dicts(
            "flow_plant_dc",
            ((i, j) for i in range(inst.num_plants) for j in range(inst.num_dcs)),
            lowBound=0.0,
            cat=pulp.LpContinuous
        )
        w = pulp.LpVariable.dicts(
            "flow_dc_cust",
            ((j, k) for j in range(inst.num_dcs) for k in range(inst.num_customers)),
            lowBound=0.0,
            cat=pulp.LpContinuous
        )

        # Dual variables for Bertsimas-Sim robust counterpart per DC
        z = pulp.LpVariable.dicts("robust_z", range(inst.num_dcs), lowBound=0.0, cat=pulp.LpContinuous)
        p = pulp.LpVariable.dicts(
            "robust_p",
            ((j, k) for j in range(inst.num_dcs) for k in range(inst.num_customers)),
            lowBound=0.0,
            cat=pulp.LpContinuous
        )

        d_bar = np.array(inst.customer_demands, dtype=float)
        d_hat = d_bar * self.uncertainty_factor

        # Objective Function
        fixed_cost = pulp.lpSum(inst.dc_opening_costs[j] * y[j] for j in range(inst.num_dcs))
        plant_transport = pulp.lpSum(
            inst.cost_plant_dc[i, j] * x[i, j]
            for i in range(inst.num_plants)
            for j in range(inst.num_dcs)
        )
        cust_transport = pulp.lpSum(
            inst.cost_dc_cust[j, k] * w[j, k]
            for j in range(inst.num_dcs)
            for k in range(inst.num_customers)
        )

        prob += fixed_cost + plant_transport + cust_transport, "Total_Robust_Cost"

        # 1. Demand Satisfaction for Nominal Demand
        for k in range(inst.num_customers):
            prob += pulp.lpSum(w[j, k] for j in range(inst.num_dcs)) == d_bar[k], f"Nominal_Demand_Cust_{k}"

        # 2. Plant Production Capacity
        for i in range(inst.num_plants):
            prob += pulp.lpSum(x[i, j] for j in range(inst.num_dcs)) <= inst.plant_capacities[i], f"Plant_Cap_{i}"

        # 3. Robust Flow Conservation and DC Capacity with Gamma-Budget Duality
        for j in range(inst.num_dcs):
            inflow = pulp.lpSum(x[i, j] for i in range(inst.num_plants))
            outflow = pulp.lpSum(w[j, k] for k in range(inst.num_customers))
            prob += inflow == outflow, f"Conservation_DC_{j}"

            # Robust Dual Bounds: z_j + p_jk >= (d_hat_k / d_bar_k) * w_jk
            for k in range(inst.num_customers):
                ratio = d_hat[k] / max(1e-4, d_bar[k])
                prob += z[j] + p[j, k] >= ratio * w[j, k], f"Robust_Dual_{j}_{k}"

            # Robust DC Capacity constraint with uncertainty buffer
            uncertainty_buffer = self.gamma * z[j] + pulp.lpSum(p[j, k] for k in range(inst.num_customers))
            prob += outflow + uncertainty_buffer <= inst.dc_capacities[j] * y[j], f"Robust_DC_Cap_{j}"

        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_sec, msg=False)
        prob.solve(solver)

        opened_dcs = [j for j in range(inst.num_dcs) if pulp.value(y[j]) is not None and pulp.value(y[j]) > 0.5]
        total_cost = float(pulp.value(prob.objective) or 0.0)
        fixed_c = float(sum(inst.dc_opening_costs[j] for j in opened_dcs))

        return {
            "status": pulp.LpStatus[prob.status],
            "robust_cost": total_cost,
            "fixed_cost": fixed_c,
            "transport_cost": total_cost - fixed_c,
            "opened_dcs": opened_dcs,
            "num_opened": len(opened_dcs),
            "gamma": float(self.gamma),
            "uncertainty_factor": self.uncertainty_factor
        }
