import gurobipy as gp
from gurobipy import GRB

def maximize_epsilon_integer_scaled():
    # Create a new Gurobi model
    m = gp.Model("framestack_max_epsilon_integer_scaled")
    m.Params.OutputFlag = 0  # Disable solver output

    # ── 1. Fixed slack ─────────────────────
    eps_slack = 1e-6

    # ── 2. Decision variables ─────────────────────────
    epsilon = m.addVar(lb=eps_slack, name="epsilon")  # To maximize

    # Scaled integer variables
    alpha_scaled = m.addVar(vtype=GRB.INTEGER, name="alpha_scaled")
    beta_scaled  = m.addVar(vtype=GRB.INTEGER, name="beta_scaled")
    gamma_scaled = m.addVar(vtype=GRB.INTEGER, name="gamma_scaled")
    delta_scaled = m.addVar(vtype=GRB.INTEGER, name="delta_scaled")
    T_scaled     = m.addVar(vtype=GRB.INTEGER, name="T_scaled")

    # Actual real variables (linked to scaled vars via epsilon)
    alpha = m.addVar(lb=0.0, name="alpha")
    beta  = m.addVar(lb=0.0, name="beta")
    gamma = m.addVar(lb=0.0, name="gamma")
    delta = m.addVar(lb=0.0, name="delta")
    T     = m.addVar(lb=0.0, name="T")

    # ── 3. Linking constraints ─────────────────────────
    m.addConstr(alpha == alpha_scaled * epsilon, name="link_alpha")
    m.addConstr(beta  == beta_scaled  * epsilon, name="link_beta")
    m.addConstr(gamma == gamma_scaled * epsilon, name="link_gamma")
    m.addConstr(delta == delta_scaled * epsilon, name="link_delta")
    m.addConstr(T     == T_scaled     * epsilon, name="link_T")

    # ── 4. Constraint parameters ──────────────────────
    non_urgent_lower_bound = 0.8
    urgent_ub = 0.3

    at_entry = 0.0
    at_exit = 0.25
    non_linked = 0.5
    at_other = 1.0

    # ── 5. Constraints (all linear) ───────────────────
    m.addConstr((non_urgent_lower_bound * alpha) + epsilon + at_other * beta >= T + eps_slack, name="c1")
    m.addConstr((non_urgent_lower_bound * alpha) + epsilon + non_linked * beta + delta >= T + eps_slack, name="c2")
    m.addConstr((urgent_ub * alpha + epsilon) + at_other * beta + gamma + delta <= T, name="c3")
    m.addConstr(alpha + epsilon + non_linked * beta + gamma <= T, name="c4")
    m.addConstr((urgent_ub * alpha + epsilon) + at_other * beta + gamma <= (non_urgent_lower_bound * alpha), name="c5")
    m.addConstr(gamma <= (urgent_ub * alpha + epsilon) + at_exit * beta, name="c6")
    m.addConstr(at_exit * beta + gamma + epsilon <= (urgent_ub * alpha) + non_linked * beta, name="c7")

    # ── 6. Objective ─────────────────────────────────
    m.setObjective(epsilon, GRB.MAXIMIZE)

    # ── 7. Solve ─────────────────────────────────────
    m.optimize()

    # ── 8. Report ────────────────────────────────────
    if m.status == GRB.OPTIMAL:
        e = epsilon.X
        print(f"Optimal ε = {e:.8f} → scaling factor = {1/e:.6f}")

        print("\n--- Raw values ---")
        print(f"α = {alpha.X:.6f}, β = {beta.X:.6f}, γ = {gamma.X:.6f}, δ = {delta.X:.6f}, ε = {e:.6f}, T = {T.X:.6f}")

        print("\n--- Integer-scaled values ---")
        print(f"α_scaled = {alpha_scaled.X:.0f}")
        print(f"β_scaled = {beta_scaled.X:.0f}, γ_scaled = {gamma_scaled.X:.0f}, δ_scaled = {delta_scaled.X:.0f}, ε_scaled = {e / e:.0f}, T_scaled = {T_scaled.X:.0f}")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    maximize_epsilon_integer_scaled()
