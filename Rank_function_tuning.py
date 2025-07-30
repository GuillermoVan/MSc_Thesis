import gurobipy as gp
from gurobipy import GRB

def maximize_epsilon_integer_scaled():
    # Create a new Gurobi model
    m = gp.Model("framestack_max_epsilon_integer_scaled")
    m.Params.OutputFlag = 0  # Disable solver output

    # ── 1. Fixed alpha and slack ─────────────────────
    alpha_val = 1.0
    eps_slack = 1e-6

    # ── 2. Decision variables ─────────────────────────
    # Continuous positive variable to maximize
    epsilon = m.addVar(lb=eps_slack, name="epsilon")

    # Scaled variables: integers
    beta_scaled  = m.addVar(vtype=GRB.INTEGER, name="beta_scaled")
    gamma_scaled = m.addVar(vtype=GRB.INTEGER, name="gamma_scaled")
    delta_scaled = m.addVar(vtype=GRB.INTEGER, name="delta_scaled")
    T_scaled     = m.addVar(vtype=GRB.INTEGER, name="T_scaled")

    # Actual variables: linked to epsilon
    beta  = m.addVar(lb=0.0, name="beta")
    gamma = m.addVar(lb=0.0, name="gamma")
    delta = m.addVar(lb=0.0, name="delta")
    T     = m.addVar(lb=0.0, name="T")

    # Link scaled variables to actual variables via epsilon
    m.addConstr(beta  == beta_scaled  * epsilon, name="link_beta")
    m.addConstr(gamma == gamma_scaled * epsilon, name="link_gamma")
    m.addConstr(delta == delta_scaled * epsilon, name="link_delta")
    m.addConstr(T     == T_scaled     * epsilon, name="link_T")

    # ── 3. Constraints ────────────────────────────────
    m.addConstr(0.85 * alpha_val + beta >= T + eps_slack, name="c1")
    m.addConstr(0.85 * alpha_val + 0.5 * beta + delta >= T + eps_slack, name="c2")
    m.addConstr(0.1 * alpha_val + beta + gamma + delta + epsilon <= T, name="c3")
    m.addConstr(1.0 * alpha_val + 0.5 * beta + gamma + epsilon <= T, name="c4")
    m.addConstr(0.1 * alpha_val + beta + gamma + epsilon <= 0.85 * alpha_val, name="c5")
    m.addConstr(gamma <= 0.1 * alpha_val + 0.25 * beta, name="c6")
    m.addConstr(0.25 * beta + gamma + epsilon <= 0.1 * alpha_val + 0.5 * beta, name="c7")

    # ── 4. Objective ─────────────────────────────────
    m.setObjective(epsilon, GRB.MAXIMIZE)

    # ── 5. Solve ─────────────────────────────────────
    m.optimize()

    # ── 6. Report ────────────────────────────────────
    if m.status == GRB.OPTIMAL:
        e = epsilon.X
        print(f"Optimal ε = {e:.8f} → scaling factor = {1/e:.6f}")
        print("\n--- Raw values ---")
        print(f"α = {alpha_val:.6f}, β = {beta.X:.6f}, γ = {gamma.X:.6f}, δ = {delta.X:.6f}, ε = {e:.6f}, T = {T.X:.6f}")

        print("\n--- Integer-scaled values ---")
        print(f"α_scaled = {alpha_val / e:.0f}")
        print(f"β_scaled = {beta_scaled.X:.0f}, γ_scaled = {gamma_scaled.X:.0f}, δ_scaled = {delta_scaled.X:.0f}, ε_scaled = {e / e:.0f}, T_scaled = {T_scaled.X:.0f}")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    maximize_epsilon_integer_scaled()
