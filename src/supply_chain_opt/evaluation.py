import pulp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .data import SupplyChainNetworkInstance

class OutOfSampleEvaluator:
    """
    Out-of-Sample Empirical Evaluator for Supply Chain Network Designs.

    Tests the operational resilience of fixed facility opening decisions y_j
    against a large unseen sample of demand shocks (e.g. 50-100 out-of-sample scenarios).

    For each test scenario, facility opening is locked (fixed first-stage investment),
    and only transportation flow is optimized to satisfy the realized demand.
    """
    def __init__(
        self,
        instance: SupplyChainNetworkInstance,
        num_oos_scenarios: int = 50,
        demand_volatility: float = 0.30,
        unmet_demand_penalty: float = 500.0,
        seed: int = 999
    ):
        self.instance = instance
        self.num_oos_scenarios = num_oos_scenarios
        self.demand_volatility = demand_volatility
        self.unmet_demand_penalty = unmet_demand_penalty
        self.seed = seed
        self.oos_scenarios = self._generate_oos_scenarios()

    def _generate_oos_scenarios(self) -> List[List[float]]:
        rng = np.random.default_rng(self.seed)
        base = np.array(self.instance.customer_demands, dtype=float)
        sigma = self.demand_volatility
        mu = -0.5 * (sigma ** 2)

        scenarios = []
        for _ in range(self.num_oos_scenarios):
            shocks = rng.lognormal(mean=mu, sigma=sigma, size=len(base))
            # Inject occasional tail demand spike (5% probability)
            tail_spike = rng.choice([1.0, 1.4], p=[0.95, 0.05], size=len(base))
            realized = [float(np.round(d * s * t, 2)) for d, s, t in zip(base, shocks, tail_spike)]
            scenarios.append(realized)
        return scenarios

    def evaluate_facility_placement(self, opened_dcs: List[int]) -> Dict[str, Any]:
        inst = self.instance
        if not opened_dcs:
            return {
                "mean_cost": float("inf"),
                "worst_cost": float("inf"),
                "p95_cost": float("inf"),
                "failure_rate": 1.0,
                "scenario_costs": []
            }

        fixed_cost = sum(inst.dc_opening_costs[j] for j in opened_dcs)
        scenario_costs = []
        unmet_events = 0

        for demands in self.oos_scenarios:
            prob = pulp.LpProblem("OOS_Subproblem", pulp.LpMinimize)
            x = pulp.LpVariable.dicts(
                "x", ((i, j) for i in range(inst.num_plants) for j in opened_dcs), lowBound=0.0
            )
            w = pulp.LpVariable.dicts(
                "w", ((j, k) for j in opened_dcs for k in range(inst.num_customers)), lowBound=0.0
            )
            unmet = pulp.LpVariable.dicts("unmet", range(inst.num_customers), lowBound=0.0)

            # Objective: transport cost + severe penalty for unmet demand
            prob += (
                pulp.lpSum(inst.cost_plant_dc[i, j] * x[i, j] for i in range(inst.num_plants) for j in opened_dcs) +
                pulp.lpSum(inst.cost_dc_cust[j, k] * w[j, k] for j in opened_dcs for k in range(inst.num_customers)) +
                pulp.lpSum(self.unmet_demand_penalty * unmet[k] for k in range(inst.num_customers))
            )

            for k in range(inst.num_customers):
                prob += pulp.lpSum(w[j, k] for j in opened_dcs) + unmet[k] == demands[k]

            for i in range(inst.num_plants):
                prob += pulp.lpSum(x[i, j] for j in opened_dcs) <= inst.plant_capacities[i]

            for j in opened_dcs:
                outflow = pulp.lpSum(w[j, k] for k in range(inst.num_customers))
                prob += pulp.lpSum(x[i, j] for i in range(inst.num_plants)) == outflow
                prob += outflow <= inst.dc_capacities[j]

            pulp.PULP_CBC_CMD(msg=False).solve(prob)

            trans_cost = float(pulp.value(prob.objective) or 0.0)
            total_scen_cost = fixed_cost + trans_cost
            scenario_costs.append(total_scen_cost)

            total_unmet = sum(pulp.value(unmet[k]) or 0.0 for k in range(inst.num_customers))
            if total_unmet > 1e-3:
                unmet_events += 1

        costs_arr = np.array(scenario_costs, dtype=float)
        return {
            "num_opened": len(opened_dcs),
            "mean_cost": float(np.mean(costs_arr)),
            "std_cost": float(np.std(costs_arr)),
            "min_cost": float(np.min(costs_arr)),
            "median_cost": float(np.median(costs_arr)),
            "p95_cost": float(np.percentile(costs_arr, 95)),
            "worst_cost": float(np.max(costs_arr)),
            "failure_rate": float(unmet_events / len(self.oos_scenarios)),
            "scenario_costs": scenario_costs
        }
