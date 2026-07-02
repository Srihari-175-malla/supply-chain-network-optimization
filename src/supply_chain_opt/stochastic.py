import pulp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .data import SupplyChainNetworkInstance

class TwoStageStochasticCFLPSolver:
    """
    Two-Stage Stochastic Programming Model for Facility Location under Demand Uncertainty.

    Stage 1: Decide binary facility opening decisions y_j before demand realization.
    Stage 2: Adapt plant production x_ij^s and customer delivery flows w_jk^s across
             stochastic demand realizations s in S.

    Objective:
      Minimize: Fixed Opening Cost + E_s [ Transport Cost(s) ]
    """
    def __init__(
        self,
        instance: SupplyChainNetworkInstance,
        num_scenarios: int = 15,
        demand_volatility: float = 0.25,
        seed: int = 42
    ):
        self.instance = instance
        self.num_scenarios = num_scenarios
        self.demand_volatility = demand_volatility
        self.seed = seed
        self.scenarios = self._generate_demand_scenarios()

    def _generate_demand_scenarios(self) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        scenarios = []
        prob = 1.0 / self.num_scenarios

        base_demands = np.array(self.instance.customer_demands, dtype=float)
        sigma = self.demand_volatility
        mu = -0.5 * (sigma ** 2)

        for s in range(self.num_scenarios):
            # Log-normal multiplicative shocks with unit expectation
            shocks = rng.lognormal(mean=mu, sigma=sigma, size=len(base_demands))
            realized_demands = [float(np.round(d * s_val, 2)) for d, s_val in zip(base_demands, shocks)]
            scenarios.append({
                "scenario_id": s,
                "probability": prob,
                "demands": realized_demands,
                "total_demand": float(sum(realized_demands))
            })
        return scenarios

    def solve(self, time_limit_sec: int = 30) -> Dict[str, Any]:
        inst = self.instance
        prob = pulp.LpProblem("TwoStage_Stochastic_CFLP", pulp.LpMinimize)

        # Stage 1: Facility Opening Variables
        y = pulp.LpVariable.dicts("open_dc", range(inst.num_dcs), cat=pulp.LpBinary)

        # Stage 2: Flow Variables per Scenario
        x = pulp.LpVariable.dicts(
            "flow_plant_dc",
            ((s, i, j)
             for s in range(self.num_scenarios)
             for i in range(inst.num_plants)
             for j in range(inst.num_dcs)),
            lowBound=0.0,
            cat=pulp.LpContinuous
        )
        w = pulp.LpVariable.dicts(
            "flow_dc_cust",
            ((s, j, k)
             for s in range(self.num_scenarios)
             for j in range(inst.num_dcs)
             for k in range(inst.num_customers)),
            lowBound=0.0,
            cat=pulp.LpContinuous
        )

        # First-stage fixed cost
        stage1_cost = pulp.lpSum(inst.dc_opening_costs[j] * y[j] for j in range(inst.num_dcs))

        # Expected second-stage transport cost
        expected_stage2_cost = pulp.lpSum(
            sc["probability"] * (
                pulp.lpSum(inst.cost_plant_dc[i, j] * x[s, i, j]
                           for i in range(inst.num_plants) for j in range(inst.num_dcs)) +
                pulp.lpSum(inst.cost_dc_cust[j, k] * w[s, j, k]
                           for j in range(inst.num_dcs) for k in range(inst.num_customers))
            )
            for s, sc in enumerate(self.scenarios)
        )

        prob += stage1_cost + expected_stage2_cost, "Expected_Total_Cost"

        # Constraints for each scenario s
        for s, sc in enumerate(self.scenarios):
            demands = sc["demands"]

            # Customer demand satisfaction in scenario s
            for k in range(inst.num_customers):
                prob += (
                    pulp.lpSum(w[s, j, k] for j in range(inst.num_dcs)) == demands[k],
                    f"Scen_{s}_Demand_Cust_{k}"
                )

            # Plant capacity in scenario s
            for i in range(inst.num_plants):
                prob += (
                    pulp.lpSum(x[s, i, j] for j in range(inst.num_dcs)) <= inst.plant_capacities[i],
                    f"Scen_{s}_Plant_Cap_{i}"
                )

            # Flow conservation & DC capacity in scenario s
            for j in range(inst.num_dcs):
                inflow = pulp.lpSum(x[s, i, j] for i in range(inst.num_plants))
                outflow = pulp.lpSum(w[s, j, k] for k in range(inst.num_customers))
                prob += inflow == outflow, f"Scen_{s}_Conservation_DC_{j}"
                prob += outflow <= inst.dc_capacities[j] * y[j], f"Scen_{s}_DC_Cap_{j}"

        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_sec, msg=False)
        prob.solve(solver)

        opened_dcs = [j for j in range(inst.num_dcs) if pulp.value(y[j]) is not None and pulp.value(y[j]) > 0.5]
        total_exp_cost = float(pulp.value(prob.objective) or 0.0)
        fixed_c = float(sum(inst.dc_opening_costs[j] for j in opened_dcs))
        exp_trans_c = total_exp_cost - fixed_c

        # Compute scenario costs
        scenario_costs = []
        for s, sc in enumerate(self.scenarios):
            s_trans = sum(
                inst.cost_plant_dc[i, j] * (pulp.value(x[s, i, j]) or 0.0)
                for i in range(inst.num_plants) for j in range(inst.num_dcs)
            ) + sum(
                inst.cost_dc_cust[j, k] * (pulp.value(w[s, j, k]) or 0.0)
                for j in range(inst.num_dcs) for k in range(inst.num_customers)
            )
            scenario_costs.append(fixed_c + s_trans)

        return {
            "status": pulp.LpStatus[prob.status],
            "expected_cost": total_exp_cost,
            "fixed_cost": fixed_c,
            "expected_transport_cost": exp_trans_c,
            "opened_dcs": opened_dcs,
            "num_opened": len(opened_dcs),
            "num_scenarios": self.num_scenarios,
            "scenario_costs": scenario_costs,
            "worst_scenario_cost": max(scenario_costs) if scenario_costs else total_exp_cost,
            "best_scenario_cost": min(scenario_costs) if scenario_costs else total_exp_cost,
        }
