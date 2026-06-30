import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class SupplyChainNetworkInstance:
    """
    Multi-Echelon Supply Chain Network Instance (Plants -> DCs -> Customers).
    
    Attributes:
        num_plants: Number of supply manufacturing plants.
        num_dcs: Number of candidate distribution center locations.
        num_customers: Number of demand nodes / customer zones.
        plant_coords: (x, y) coordinates of production plants.
        dc_coords: (x, y) coordinates of candidate distribution centers.
        cust_coords: (x, y) coordinates of customer zones.
        plant_capacities: Maximum production output per plant.
        dc_capacities: Maximum throughput capacity per DC.
        dc_opening_costs: Fixed capital expenditure to commission/open each DC.
        customer_demands: Base nominal demand per customer zone.
        min_dc_throughput: Minimum throughput required if a DC is opened (operational threshold).
        max_service_dist: Maximum allowable distance between DC and customer (service-level agreement).
    """
    def __init__(
        self,
        num_plants: int,
        num_dcs: int,
        num_customers: int,
        plant_coords: List[Tuple[float, float]],
        dc_coords: List[Tuple[float, float]],
        cust_coords: List[Tuple[float, float]],
        plant_capacities: List[float],
        dc_capacities: List[float],
        dc_opening_costs: List[float],
        customer_demands: List[float],
        min_dc_throughput: float = 0.0,
        max_service_dist: float = 1000.0,
        plant_unit_transport_cost: float = 1.0,
        dc_unit_transport_cost: float = 1.5
    ):
        self.num_plants = num_plants
        self.num_dcs = num_dcs
        self.num_customers = num_customers

        self.plant_coords = plant_coords
        self.dc_coords = dc_coords
        self.cust_coords = cust_coords

        self.plant_capacities = plant_capacities
        self.dc_capacities = dc_capacities
        self.dc_opening_costs = dc_opening_costs
        self.customer_demands = customer_demands

        self.min_dc_throughput = min_dc_throughput
        self.max_service_dist = max_service_dist

        self.cost_plant_dc = self._compute_cost_matrix(plant_coords, dc_coords, unit_rate=plant_unit_transport_cost)
        self.cost_dc_cust = self._compute_cost_matrix(dc_coords, cust_coords, unit_rate=dc_unit_transport_cost)

    def _compute_cost_matrix(
        self, src: List[Tuple[float, float]], dst: List[Tuple[float, float]], unit_rate: float
    ) -> np.ndarray:
        matrix = np.zeros((len(src), len(dst)), dtype=float)
        for i, (x1, y1) in enumerate(src):
            for j, (x2, y2) in enumerate(dst):
                dist = np.hypot(x1 - x2, y1 - y2)
                matrix[i, j] = dist * unit_rate
        return matrix

    @property
    def total_demand(self) -> float:
        return float(sum(self.customer_demands))

    @property
    def total_plant_capacity(self) -> float:
        return float(sum(self.plant_capacities))

    @property
    def total_dc_capacity(self) -> float:
        return float(sum(self.dc_capacities))


def generate_synthetic_supply_chain_network(
    n_plants: int = 4,
    n_dcs: int = 8,
    n_cust: int = 25,
    seed: int = 42,
    plant_cap_multiplier: float = 1.6,
    dc_cap_multiplier: float = 0.35,
    min_dc_throughput: float = 0.0,
    max_service_dist: float = 1000.0
) -> SupplyChainNetworkInstance:
    """
    Generates a reproducible synthetic multi-echelon supply chain instance.
    """
    rng = np.random.default_rng(seed)

    plant_coords = [(float(x), float(y)) for x, y in rng.uniform(0, 100, (n_plants, 2))]
    dc_coords = [(float(x), float(y)) for x, y in rng.uniform(10, 90, (n_dcs, 2))]
    cust_coords = [(float(x), float(y)) for x, y in rng.uniform(0, 100, (n_cust, 2))]

    customer_demands = [float(d) for d in rng.integers(15, 35, n_cust)]
    total_dem = sum(customer_demands)

    # Size capacities realistically relative to total demand
    plant_capacities = [float(np.round((total_dem * plant_cap_multiplier) / n_plants, 1))] * n_plants
    dc_capacities = [float(np.round(total_dem * dc_cap_multiplier, 1))] * n_dcs
    
    # Heterogeneous fixed opening costs based on central location premium
    dc_opening_costs = []
    for j, (x, y) in enumerate(dc_coords):
        dist_center = np.hypot(x - 50.0, y - 50.0)
        base_cost = 2500.0 + 150.0 * j
        location_factor = 1.0 + (50.0 - dist_center) / 100.0
        dc_opening_costs.append(float(np.round(base_cost * location_factor, 1)))

    return SupplyChainNetworkInstance(
        num_plants=n_plants,
        num_dcs=n_dcs,
        num_customers=n_cust,
        plant_coords=plant_coords,
        dc_coords=dc_coords,
        cust_coords=cust_coords,
        plant_capacities=plant_capacities,
        dc_capacities=dc_capacities,
        dc_opening_costs=dc_opening_costs,
        customer_demands=customer_demands,
        min_dc_throughput=min_dc_throughput,
        max_service_dist=max_service_dist
    )


def create_cflp_benchmark_instance(instance_name: str = "cap71") -> SupplyChainNetworkInstance:
    """
    Creates a benchmark instance parameterized after standard OR-Library CFLP instances.
    """
    if instance_name == "cap71":
        # 4 plants, 8 candidate DCs, 30 customers
        return generate_synthetic_supply_chain_network(n_plants=4, n_dcs=8, n_cust=30, seed=101)
    elif instance_name == "cap101":
        # 5 plants, 12 candidate DCs, 50 customers
        return generate_synthetic_supply_chain_network(n_plants=5, n_dcs=12, n_cust=50, seed=202)
    elif instance_name == "cap131":
        # 6 plants, 16 candidate DCs, 80 customers
        return generate_synthetic_supply_chain_network(n_plants=6, n_dcs=16, n_cust=80, seed=303)
    else:
        return generate_synthetic_supply_chain_network(n_plants=3, n_dcs=6, n_cust=20, seed=42)
