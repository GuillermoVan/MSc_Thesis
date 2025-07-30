import gurobipy as gp
from gurobipy import GRB

def maximize_epsilon_full():
    m = gp.Model("loop_supply_max_epsilon")
    m.Params.OutputFlag = 0  # Cleaner output

    # Fixed α value
    alpha_val = 1.0

    # Decision variables
    beta    = m.addVar(lb=0.0, name="beta")
    gamma   = m.addVar(lb=0.0, name="gamma")
    delta   = m.addVar(lb=0.0, name="delta")
    epsilon = m.addVar(lb=1e-6, name="epsilon")  # ε > 0
    T       = m.addVar(lb=0.0, name="T")

    # ────────────────────────────────────────────────────────
    # NEVER-ALLOW constraints (bad totes must exceed T by ε)
    # ────────────────────────────────────────────────────────

    # C1: Released, non-urgent, not on doorstep
    # f(psg)=0.85, f(l)=1, f(d)=0, f(vcap)=0
    m.addConstr(0.85 * alpha_val + beta >= T + epsilon, name="c1_never")

    # C2: Non-started, non-urgent, over vcap
    # f(psg)=0.85, f(l)=0.5, f(d)=0, f(vcap)=1
    m.addConstr(0.85 * alpha_val + 0.5 * beta + delta >= T + epsilon, name="c2_never")

    # ────────────────────────────────────────────────────────
    # ALWAYS-ALLOW constraints (good totes must fit under T)
    # ────────────────────────────────────────────────────────

    # C3: Urgent tote (f(psg)=0.1, others = 1)
    m.addConstr(0.1 * alpha_val + beta + gamma + delta + epsilon <= T, name="c3_always")

    # C4: Non-started, under capacity (f(psg)=1.0, f(l)=0.5, f(d)=1, f(vcap)=0)
    m.addConstr(1.0 * alpha_val + 0.5 * beta + gamma + epsilon <= T, name="c4_always")

    # ────────────────────────────────────────────────────────
    # PRIORITIZATION constraints (A should score better than B)
    # All include ε on the LHS to create margin of improvement
    # ────────────────────────────────────────────────────────

    # C5: Urgent, non-doorstep < Non-urgent, doorstep
    # LHS (urgent) = 0.1α + β + γ + ε
    # RHS (non-urgent doorstep) = 0.85α
    m.addConstr(0.1 * alpha_val + beta + gamma + epsilon <= 0.85 * alpha_val, name="c5_priority")

    # C6: Entry doorstep < Exit doorstep
    # LHS: 0.1α + γ + ε, RHS: 0.1α + 0.25β
    m.addConstr(0.1 * alpha_val + gamma + epsilon <= 0.1 * alpha_val + 0.25 * beta, name="c6_priority")

    # C7: Exit doorstep < Non-started
    # LHS: 0.1α + 0.25β + γ + ε, RHS: 0.1α + 0.5β
    m.addConstr(0.1 * alpha_val + 0.25 * beta + gamma + epsilon <= 0.1 * alpha_val + 0.5 * beta, name="c7_priority")

    # ────────────────────────────────────────────────────────
    # Objective: maximize ε (urgency buffer)
    # ────────────────────────────────────────────────────────
    m.setObjective(epsilon, GRB.MAXIMIZE)

    # Solve
    m.optimize()

    # Report results
    if m.status == GRB.OPTIMAL:
        e = epsilon.X
        scale = 1 / e
        print(f"\n✅ Optimal ε = {e:.6f} (scale factor = {scale:.2f})\n")

        print("--- Raw Coefficients ---")
        print(f"α = {alpha_val:.6f}")
        print(f"β = {beta.X:.6f}")
        print(f"γ = {gamma.X:.6f}")
        print(f"δ = {delta.X:.6f}")
        print(f"T = {T.X:.6f}")
        print(f"ε = {e:.6f}")

        print("\n--- Scaled Coefficients ---")
        print(f"α_scaled = {alpha_val * scale:.0f}")
        print(f"β_scaled = {beta.X * scale:.0f}")
        print(f"γ_scaled = {gamma.X * scale:.0f}")
        print(f"δ_scaled = {delta.X * scale:.0f}")
        print(f"T_scaled = {T.X * scale:.0f}")
        print(f"ε_scaled = {e * scale:.0f}")

        print("\n--- Constraint Slacks ---")
        for c in m.getConstrs():
            print(f"{c.ConstrName}: slack = {c.Slack:.6e}, dual = {c.Pi:.6e}")
    else:
        print("❌ No optimal solution found.")

if __name__ == "__main__":
    maximize_epsilon_full()
