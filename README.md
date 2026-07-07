# Supply Chain Network Design & Capacitated Facility Location Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Optimization: PuLP / CBC / HiGHS](https://img.shields.io/badge/Solvers-PuLP%20%7C%20CBC%20%7C%20HiGHS-green.svg)](https://coin-or.github.io/pulp/)
[![Tests: PyTest](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)](tests/)

A mathematical optimization and operations research platform in Python implementing **Multi-Echelon Capacitated Facility Location (CFLP)**, **Benders Decomposition**, **Two-Stage Stochastic Programming**, and **Bertsimas-Sim ($\Gamma$-Budget) Robust Optimization** with empirical out-of-sample stress testing.

---

## 📌 Architecture & Mathematical Formulations

```
Manufacturing Plants (i ∈ I)
       │  [Production Capacity P_i, Unit Cost c_ij]
       ▼
Candidate Distribution Centers (j ∈ J)  ──▶ [Binary Opening y_j ∈ {0,1}, Capacity C_j, Fixed Cost f_j]
       │  [Flow Conservation & Min-Throughput M_j, Unit Cost g_jk]
       ▼
Customer Demand Zones (k ∈ K)           ──▶ [Demand d_k (Deterministic / Stochastic / Robust)]
```

### 1. Monolithic Multi-Echelon CFLP (MILP)
$$\min_{y, x, w} \sum_{j \in J} f_j y_j + \sum_{i \in I}\sum_{j \in J} c_{ij} x_{ij} + \sum_{j \in J}\sum_{k \in K} g_{jk} w_{jk}$$

Subject to:
1. **Demand Satisfaction**: $\sum_{j \in J} w_{jk} = d_k, \quad \forall k \in K$
2. **Plant Capacity**: $\sum_{j \in J} x_{ij} \le P_i, \quad \forall i \in I$
3. **Flow Conservation**: $\sum_{i \in I} x_{ij} = \sum_{k \in K} w_{jk}, \quad \forall j \in J$
4. **DC Capacity**: $\sum_{k \in K} w_{jk} \le C_j y_j, \quad \forall j \in J$
5. **Operational Min-Throughput**: $\sum_{k \in K} w_{jk} \ge M_j y_j, \quad \forall j \in J$

---

### 2. Benders Decomposition Algorithm
Decomposes the combinatorial facility investment decision from continuous multi-commodity transportation routing:
- **Master Problem (MIP)**: Chooses binary facility selection $y_j \in \{0, 1\}$ and auxiliary epigraph estimator $\eta$.
- **Subproblem (LP)**: Computes minimum-cost network flow given fixed $\bar{y}$.
- **Cuts Generated**:
  - *Feasibility Cuts*: Enforces total open capacity coverage when subproblem is infeasible.
  - *Optimality Cuts*: $\eta \ge \text{SubCost}(\bar{y}) \cdot \left(\frac{\sum_{j \in \text{open}} y_j}{|\text{open}|}\right)$.

---

### 3. Two-Stage Stochastic Programming
Optimizes expected total cost over a discrete set of scenarios $S$ with probabilities $p_s$:
$$\min_{y \in \{0,1\}^{|J|}} \sum_{j \in J} f_j y_j + \sum_{s \in S} p_s \left( \sum_{i \in I}\sum_{j \in J} c_{ij} x_{ij}^s + \sum_{j \in J}\sum_{k \in K} g_{jk} w_{jk}^s \right)$$

---

### 4. Bertsimas-Sim ($\Gamma$-Budget) Robust Optimization
Protects against interval demand uncertainty $d_k \in [\bar{d}_k, \bar{d}_k + \hat{d}_k]$ where at most $\Gamma$ customer demands deviate simultaneously:
$$\sum_{k \in K} w_{jk} + \left( \Gamma z_j + \sum_{k \in K} p_{jk} \right) \le C_j y_j, \quad z_j + p_{jk} \ge \frac{\hat{d}_k}{\bar{d}_k} w_{jk}$$

---

## 📊 Benchmark Results

Evaluated on a realistic multi-echelon network instance ($5\text{ Plants} \to 12\text{ Candidate DCs} \to 35\text{ Customer Demand Zones}$):

### Formulation Performance Summary
| Optimization Formulation | Open DCs | Solve Time (s) | Speedup vs MILP | Total Objective ($) | Cost Δ vs Deterministic |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Monolithic Multi-Echelon MILP** | 4 | 0.0824 s | 1.00x (Baseline) | $26,482.50 | Baseline |
| **2. Benders Decomposition Engine** | 4 | 0.0415 s | **1.99x** | $26,482.50 | +0.00% (Exact Optimum) |
| **3. Two-Stage Stochastic (15 Scenarios)** | 5 | 0.1280 s | 0.64x | $27,940.10 | +5.50% (Risk Premium) |
| **4. Robust Opt ($\Gamma$-Budget)** | 5 | 0.0650 s | 1.27x | $28,310.80 | +6.90% (Worst-Case Buffer) |

### Out-of-Sample Empirical Stress Test (50 Unseen Realizations)
| Placement Strategy | Mean OOS Cost ($) | P95 OOS Cost ($) | Worst-Case Cost ($) | Failure / Shortage Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Deterministic MILP** | $28,140.20 | $34,850.00 | $38,920.00 | 8.0% |
| **Two-Stage Stochastic** | $27,820.40 | $31,100.00 | $32,450.00 | **0.0%** |
| **Robust Counterpart** | $28,050.10 | $30,940.00 | **$31,800.00** | **0.0%** |

> **Key Takeaway**: While the deterministic MILP achieves the lowest in-sample nominal cost, it suffers an 8% capacity shortage rate during severe demand surges. Two-Stage Stochastic and Robust optimization eliminate outages entirely and reduce worst-case out-of-sample costs by **18.3%**.

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Srihari-175-malla/supply-chain-network-optimization.git
cd supply-chain-network-optimization
pip install -r requirements.txt
pip install -e .
```

### Running the Full Benchmark Suite
```bash
python benchmarks/run_benchmark.py
```

### Running Unit Tests
```bash
python -m pytest tests/ -v
```

---

## 💻 Python API Example

```python
from supply_chain_opt.data import generate_synthetic_supply_chain_network
from supply_chain_opt.milp import MultiEchelonCFLPSolver
from supply_chain_opt.benders import BendersDecompositionSolver

# 1. Generate network instance
network = generate_synthetic_supply_chain_network(n_plants=4, n_dcs=8, n_cust=25)

# 2. Solve with Monolithic MILP
milp_solver = MultiEchelonCFLPSolver(network, single_sourcing=False)
milp_result = milp_solver.solve()
print(f"Optimal DCs Opened: {milp_result['opened_dcs']} | Total Cost: ${milp_result['total_cost']:,.2f}")

# 3. Solve with Benders Decomposition
benders_solver = BendersDecompositionSolver(network)
benders_result = benders_solver.solve()
print(f"Benders Iterations: {benders_result['iterations']} | Cost: ${benders_result['total_cost']:,.2f}")
```

---

## 📂 Repository Structure

```
.
├── benchmarks/
│   └── run_benchmark.py          # Empirical scalability & out-of-sample evaluation
├── src/
│   └── supply_chain_opt/
│       ├── __init__.py           # Package exports
│       ├── data.py               # Network data structures & synthetic generator
│       ├── milp.py               # Monolithic multi-echelon CFLP solver
│       ├── benders.py            # Benders decomposition master-subproblem engine
│       ├── stochastic.py         # Two-stage stochastic programming under demand shocks
│       ├── robust.py             # Bertsimas-Sim Gamma-budget robust optimization
│       ├── evaluation.py         # Out-of-sample Monte Carlo stress testing
│       └── visualization.py      # Matplotlib network flow topology visualizer
├── tests/
│   ├── test_supply_chain.py      # Comprehensive pytest test suite
├── Makefile                      # Build and test shortcuts
├── pyproject.toml                # Package configuration
└── requirements.txt              # Core dependencies
```

---

## 📜 License
MIT License. Authored by Srihari Malla (srihari175@gmail.com).
