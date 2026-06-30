import pulp
from typing import Dict, List, Tuple, Any, Optional
from .data import SupplyChainNetworkInstance

class MultiEchelonCFLPSolver:
    """
    Monolithic Multi-Echelon Capacitated Facility Location MILP Solver.
    
    Formulation:
      Minimize:
        sum_j (f_j * y_j) + sum_ij (c_ij * x_ij) + sum_jk (g_jk * w_jk)
      Subject to:
        1. sum_j w_jk == d_k                      (Demand Satisfaction)
        2. sum_j x_ij <= P_i                      (Plant Production Capacity)
        3. sum_i x_ij == sum_k w_jk               (Flow Conservation at DC j)
        4. sum_k w_jk <= C_j * y_j                (DC Throughput Capacity)
        5. sum_k w_jk >= M_j * y_j                (Minimum DC Throughput Threshold)
        6. w_jk == 0 if dist(j, k) > max_dist     (Service Level Radius Cap)
        7. y_j in {0, 1}, x_ij >= 0, w_jk >= 0 (or w_jk binary if single_sourcing)
    """
    def __init__(self, instance: SupplyChainNetworkInstance, single_sourcing: bool = False):
        self.instance = instance
        self.single_sourcing = single_sourcing

    def solve(self, time_limit_sec: int = 30) -> Dict[str, Any]:
        inst = self.instance
        prob = pulp.LpProblem("MultiEchelon_CFLP", pulp.LpMinimize)

        # Decision Variables
        y = pulp.LpVariable.dicts("open_dc", range(inst.num_dcs), cat=pulp.LpBinary)
        x = pulp.LpVariable.dicts(
            "flow_plant_dc",
            ((i, j) for i in range(inst.num_plants) for j in range(inst.num_dcs)),
            lowBound=0.0,
            cat=pulp.LpContinuous
        )
        
        if self.single_sourcing:
            # Binary customer assignment variable: 1 if DC j serves all of customer k's demand
            w = pulp.LpVariable.dicts(
                "assign_dc_cust",
                ((j, k) for j in range(inst.num_dcs) for k in range(inst.num_customers)),
                cat=pulp.LpBinary
            )
        else:
            # Continuous flow variable from DC j to customer k
            w = pulp.LpVariable.dicts(
                "flow_dc_cust",
                ((j, k) for j in range(inst.num_dcs) for k in range(inst.num_customers)),
                lowBound=0.0,
                cat=pulp.LpContinuous
            )

        # Objective Function
        fixed_cost = pulp.lpSum(inst.dc_opening_costs[j] * y[j] for j in range(inst.num_dcs))
        plant_transport = pulp.lpSum(
            inst.cost_plant_dc[i, j] * x[i, j]
            for i in range(inst.num_plants)
            for j in range(inst.num_dcs)
        )
        
        if self.single_sourcing:
            cust_transport = pulp.lpSum(
                inst.cost_dc_cust[j, k] * inst.customer_demands[k] * w[j, k]
                for j in range(inst.num_dcs)
                for k in range(inst.num_customers)
            )
        else:
            cust_transport = pulp.lpSum(
                inst.cost_dc_cust[j, k] * w[j, k]
                for j in range(inst.num_dcs)
                for k in range(inst.num_customers)
            )

        prob += fixed_cost + plant_transport + cust_transport, "Total_Supply_Chain_Cost"

        # 1. Customer Demand Satisfaction
        for k in range(inst.num_customers):
            if self.single_sourcing:
                prob += pulp.lpSum(w[j, k] for j in range(inst.num_dcs)) == 1.0, f"Demand_Assign_Cust_{k}"
            else:
                prob += pulp.lpSum(w[j, k] for j in range(inst.num_dcs)) == inst.customer_demands[k], f"Demand_Sat_Cust_{k}"

        # 2. Plant Capacity Constraints
        for i in range(inst.num_plants):
            prob += (
                pulp.lpSum(x[i, j] for j in range(inst.num_dcs)) <= inst.plant_capacities[i],
                f"Plant_Cap_{i}"
            )

        # 3. Flow Conservation & DC Capacity Constraints
        for j in range(inst.num_dcs):
            inflow = pulp.lpSum(x[i, j] for i in range(inst.num_plants))
            if self.single_sourcing:
                outflow = pulp.lpSum(inst.customer_demands[k] * w[j, k] for k in range(inst.num_customers))
            else:
                outflow = pulp.lpSum(w[j, k] for k in range(inst.num_customers))

            prob += inflow == outflow, f"Flow_Conservation_DC_{j}"
            prob += outflow <= inst.dc_capacities[j] * y[j], f"DC_Cap_{j}"

            if inst.min_dc_throughput > 0:
                prob += outflow >= inst.min_dc_throughput * y[j], f"DC_Min_Throughput_{j}"

        # 4. Service Distance Limit (if configured)
        if inst.max_service_dist < 1000.0:
            for j in range(inst.num_dcs):
                for k in range(inst.num_customers):
                    dist = inst.cost_dc_cust[j, k] / 1.5
                    if dist > inst.max_service_dist:
                        prob += w[j, k] == 0, f"Max_Dist_Cut_DC_{j}_Cust_{k}"

        # Solve MILP
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_sec, msg=False)
        prob.solve(solver)

        opened_dcs = [j for j in range(inst.num_dcs) if pulp.value(y[j]) is not None and pulp.value(y[j]) > 0.5]
        total_cost = float(pulp.value(prob.objective) or 0.0)
        fixed_c = float(sum(inst.dc_opening_costs[j] for j in opened_dcs))
        transport_c = total_cost - fixed_c

        # Extract flows
        plant_to_dc = {}
        for i in range(inst.num_plants):
            for j in range(inst.num_dcs):
                val = pulp.value(x[i, j]) or 0.0
                if val > 1e-4:
                    plant_to_dc[(i, j)] = float(val)

        dc_to_cust = {}
        for j in range(inst.num_dcs):
            for k in range(inst.num_customers):
                val = pulp.value(w[j, k]) or 0.0
                if self.single_sourcing:
                    if val > 0.5:
                        dc_to_cust[(j, k)] = float(inst.customer_demands[k])
                else:
                    if val > 1e-4:
                        dc_to_cust[(j, k)] = float(val)

        return {
            "status": pulp.LpStatus[prob.status],
            "total_cost": total_cost,
            "fixed_cost": fixed_c,
            "transport_cost": transport_c,
            "opened_dcs": opened_dcs,
            "num_opened": len(opened_dcs),
            "plant_to_dc": plant_to_dc,
            "dc_to_cust": dc_to_cust,
        }
