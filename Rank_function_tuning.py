import gurobipy as gp
from gurobipy import GRB

def maximize_epsilon():
    # Create a new Gurobi model
    m = gp.Model("framestack_max_epsilon")

    # (Optional) turn off Gurobi’s output for a cleaner run
    m.Params.OutputFlag = 0

    # ── 1. Normalization ─────────────────────────────
    # We fix alpha = 1, so all other coefficients are "relative" to alpha.
    alpha_val = 1.0

    # Use a tiny slack to approximate strict “>” constraints
    eps_slack = 1e-6

    # ── 2. Decision variables ─────────────────────────
    # β, γ, δ, ε and T are non‐negative real variables.
    beta    = m.addVar(lb=0.0, name="beta")
    gamma   = m.addVar(lb=0.0, name="gamma")
    delta   = m.addVar(lb=0.0, name="delta")
    epsilon = m.addVar(lb=0.0, name="epsilon")
    T       = m.addVar(lb=0.0, name="T")

    m.update()

    # ── 3. Constraints ────────────────────────────────
    # (1) “Never select released, non‐urgent, not‐doorstep”:
    #      0.85·α + β  >  T
    # →   0.85*1 + beta ≥ T + eps_slack
    m.addConstr(0.85 * alpha_val + beta >= T + eps_slack, name="c1")

    # (2) “Never select non‐started, non‐urgent, exceeds vcap”:
    #      0.85·α + 0.5·β + δ  >  T
    m.addConstr(0.85 * alpha_val + 0.5 * beta + delta >= T + eps_slack, name="c2")

    # (3) “Always allow urgent”:
    #      0.1·α + β + γ + δ + ε  ≤  T
    m.addConstr(0.1 * alpha_val + beta + gamma + delta + epsilon <= T, name="c3")

    # (4) “Always allow non‐started & not exceeding vcap”:
    #      1·α + 0.5·β + γ + ε  ≤  T
    m.addConstr(1.0 * alpha_val + 0.5 * beta + gamma + epsilon <= T, name="c4")

    # (5) “Urgent‐non‐doorstep ≤ non‐urgent‐doorstep (tie‐breaker)”:
    #      β + γ + ε  ≤  0.75·α
    m.addConstr(beta + gamma + epsilon <= 0.75 * alpha_val, name="c5")

    # (7′) “Exit‐doorstep ≤ non‐started (tie‐breaker)”:
    #      γ + ε  ≤  0.05·β
    m.addConstr(gamma + epsilon <= 0.05 * beta, name="c7")

    # ── 4. Objective ─────────────────────────────────
    # Maximize epsilon
    m.setObjective(epsilon, GRB.MAXIMIZE)

    # ── 5. Solve ─────────────────────────────────────
    m.optimize()

    # ── 6. Report ────────────────────────────────────
    if m.status == GRB.OPTIMAL:
        e = epsilon.X
        e_factor = 1 / e
        print(f"Optimal ε = {e} -> {1:.6f}")
        print(f"α = {alpha_val:.6f}, β = {beta.X:.6f}, γ = {gamma.X:.6f}, δ = {delta.X:.6f}, ε = {e:.6f}, T = {T.X:.6f}")
        # Multiply first, then format:
        print(f"α_scaled = {(alpha_val * e_factor):.6f}, β_scaled = {(beta.X * e_factor):.6f}, γ_scaled = {(gamma.X * e_factor):.6f}, δ_scaled = {(delta.X * e_factor):.6f}, ε_scaled = {(e * e_factor):.6f}, T_scaled = {(T.X * e_factor):.6f}")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    maximize_epsilon()
