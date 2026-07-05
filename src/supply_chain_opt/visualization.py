import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from .data import SupplyChainNetworkInstance

def plot_supply_chain_network(
    instance: SupplyChainNetworkInstance,
    opened_dcs: List[int],
    plant_to_dc: Optional[Dict[Tuple[int, int], float]] = None,
    dc_to_cust: Optional[Dict[Tuple[int, int], float]] = None,
    title: str = "Supply Chain Network Flow Topology",
    output_path: Optional[str] = None
) -> None:
    """
    Plots the supply chain network graph showing:
      - Plants (Triangles, Red)
      - Open Distribution Centers (Squares, Green)
      - Closed Candidate DCs (Squares, Gray)
      - Customers (Circles, Blue)
      - Transportation Flow Edges (scaled line thickness)
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # 1. Plot Plant -> DC Flow Edges
    if plant_to_dc:
        max_flow = max(plant_to_dc.values()) if plant_to_dc else 1.0
        for (i, j), flow in plant_to_dc.items():
            if flow > 1e-3:
                p_x, p_y = instance.plant_coords[i]
                d_x, d_y = instance.dc_coords[j]
                lw = 0.5 + 3.0 * (flow / max_flow)
                ax.plot([p_x, d_x], [p_y, d_y], color="#e74c3c", alpha=0.4, linewidth=lw, zorder=1)

    # 2. Plot DC -> Customer Flow Edges
    if dc_to_cust:
        max_flow = max(dc_to_cust.values()) if dc_to_cust else 1.0
        for (j, k), flow in dc_to_cust.items():
            if flow > 1e-3:
                d_x, d_y = instance.dc_coords[j]
                c_x, c_y = instance.cust_coords[k]
                lw = 0.5 + 2.5 * (flow / max_flow)
                ax.plot([d_x, c_x], [d_y, c_y], color="#3498db", alpha=0.35, linewidth=lw, zorder=1)

    # 3. Plot Customer Nodes
    cust_x = [c[0] for c in instance.cust_coords]
    cust_y = [c[1] for c in instance.cust_coords]
    ax.scatter(cust_x, cust_y, c="#2980b9", marker="o", s=45, label="Customer Zone", zorder=3)

    # 4. Plot Closed DCs
    closed_dcs = [j for j in range(instance.num_dcs) if j not in opened_dcs]
    if closed_dcs:
        closed_x = [instance.dc_coords[j][0] for j in closed_dcs]
        closed_y = [instance.dc_coords[j][1] for j in closed_dcs]
        ax.scatter(closed_x, closed_y, c="#95a5a6", marker="s", s=90, alpha=0.6, label="Closed Candidate DC", zorder=2)

    # 5. Plot Open DCs
    if opened_dcs:
        open_x = [instance.dc_coords[j][0] for j in opened_dcs]
        open_y = [instance.dc_coords[j][1] for j in opened_dcs]
        ax.scatter(open_x, open_y, c="#27ae60", marker="s", s=140, edgecolors="#1e8449", linewidths=1.5, label="Opened DC", zorder=4)

    # 6. Plot Manufacturing Plants
    plant_x = [p[0] for p in instance.plant_coords]
    plant_y = [p[1] for p in instance.plant_coords]
    ax.scatter(plant_x, plant_y, c="#c0392b", marker="^", s=180, edgecolors="#922b21", linewidths=1.5, label="Manufacturing Plant", zorder=5)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X Coordinate (km)")
    ax.set_ylabel("Y Coordinate (km)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close()
