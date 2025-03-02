import gurobipy as gp
from gurobipy import GRB
import numpy as np
from model import master_problem, optimality_sub, feasibility_sub
import time

def benders(master_model, I, theta, sub_opt, sub_feas, sets, parameters, d_its, eps, thetaLB, pi_opt_inventory_balance_U, pi_opt_initial_shortage_U, pi_opt_flow_conservation, pi_opt_arc_capacity, pi_feas_inventory_balance_U, pi_feas_initial_shortage_U, pi_feas_flow_conservation, pi_feas_arc_capacity, time_limit=10800):

    # Extract required sets and parameters
    V, R, T, A, V0, V1, V3 = sets["V"], sets["R"], sets["T"], sets["A"], sets["V0"], sets["V1"], sets["V3"]
    t_ijt, p_i, cost_ir, m_ij, g_shelter, g_pod, w_r, K_ijt, P, sce = (
        parameters["t_ijt"], parameters["p_i"], parameters["cost_ir"], parameters["m_ij"],
        parameters["g_shelter"], parameters["g_pod"], parameters["w_r"], parameters["K_ijt"],
        parameters["P"], range(len(parameters["P"]))  # Number of scenarios
    )
    start_time = time.time()
     # Initialize bounds and tolerance
    LB = thetaLB
    UB = 1e10

    # Counters for optimality and feasibility cuts
    Ocuts = 0
    Fcuts = 0

    # Iteration counter
    iter = 0
    
    # Number of scenarios
    num = len(sce)

    # Store first-stage decision variable values
    I_values = {}
    theta_value = 0

    # Storage for optimality subproblem dual multipliers
    pi_opt_inventory_balance_U_val = {s: {} for s in range(num)}
    pi_opt_initial_shortage_U_val = {s: {} for s in range(num)}
    pi_opt_flow_conservation_val = {s: {} for s in range(num)}
    pi_opt_arc_capacity_val = {s: {} for s in range(num)}
    
    # Storage for feasibility subproblem dual multipliers
    pi_feas_inventory_balance_U_val = {}
    pi_feas_initial_shortage_U_val = {}
    pi_feas_flow_conservation_val = {}
    pi_feas_arc_capacity_val = {}

    # Start the Benders loop
    while (UB - LB) / max(1e-10, abs(UB)) > eps:
        iter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            print(f"\nStopping Benders after {elapsed_time:.2f} seconds (Time limit reached)")
            break  # Stop execution if the time limit is reached

        # Solve the master problem
        master_model.optimize()

        # Check if the optimization was successful
        if master_model.status == GRB.OPTIMAL:
            # Extract the values of first-stage decision variables I[i, r, t]
            for i in range(V):
                for r in range(R):
                    for t in range(T):
                        I_values[i, r, t] = I[i, r, t].x  # Store decision variable values

            # Get the value of theta
            theta_value = theta.x

            # Update the lower bound
            LB = master_model.objVal       
            
        elif master_model.status == GRB.INFEASIBLE:
            print("Master problem is infeasible.")
            exit(0)  # Or handle infeasibility appropriately

        elif master_model.status == GRB.UNBOUNDED:
            print("Master problem is unbounded.")
            exit(0)  # Or handle unboundedness appropriately

        else:
            print(f"Optimization was stopped with status {master_model.status}")
            exit(0)

        # Initialize storage for subproblem solutions
        Q_opt = np.zeros(num)  # Stores optimality subproblem objective values
        w_feas = np.zeros(num)  # Stores feasibility subproblem objective values
        FLAG = 0  # (0 if I_hat is feasible for all scenarios, 1 otherwise)
        infeasible_sce = -1  # Track which scenario is infeasible

        # Loop over all scenarios
        for s in range(num):
            # Update RHS for inventory flow balance constraints
            for i in range(V):
                for t in range(1, T):
                    pi_opt_inventory_balance_U[i, t].setAttr(GRB.Attr.RHS, d_its[(i, t, s)])

            # Update RHS for initial shortage constraints
            for i in range(V):
                pi_opt_initial_shortage_U[i].setAttr(GRB.Attr.RHS, d_its[(i, 0, s)])

            # Update RHS for flow conservation constraints
            for i in range(V):
                for r in range(R):
                    for t in range(T - 1):
                        pi_opt_flow_conservation[i, r, t].setAttr(GRB.Attr.RHS, I_values[i, r, t + 1] - I_values[i, r, t])

            # Solve the optimality subproblem
            sub_opt.optimize()

            if sub_opt.status == GRB.OPTIMAL:
                # Store the optimal objective value
                Q_opt[s] = sub_opt.objVal

                # Retrieve and store dual values for each constraint
                for key, constraint in pi_opt_inventory_balance_U.items():
                    pi_opt_inventory_balance_U_val[s][key] = constraint.pi
                for key, constraint in pi_opt_initial_shortage_U.items():
                    pi_opt_initial_shortage_U_val[s][key] = constraint.pi
                for key, constraint in pi_opt_flow_conservation.items():
                    pi_opt_flow_conservation_val[s][key] = constraint.pi
                for key, constraint in pi_opt_arc_capacity.items():
                    pi_opt_arc_capacity_val[s][key] = constraint.pi

            elif sub_opt.status == GRB.INFEASIBLE:
                # If infeasible, set FLAG to 1 and mark the scenario
                FLAG = 1
                infeasible_sce = s
                break;  # No need to continue checking other scenarios

            else:
                print(f"Error: One of the scenario subproblems is infeasible (Status: {sub_opt.status})")
                exit(0)

        print("FLAG =", FLAG)

        if FLAG == 1:
            # Solve feasibility subproblem for the detected infeasible scenario
            s = infeasible_sce

            # Update RHS for inventory flow balance constraints
            for i in range(V):
                for t in range(1, T):
                    pi_feas_inventory_balance_U[i, t].setAttr(GRB.Attr.RHS, d_its[(i, t, s)])

            # Update RHS for initial shortage constraints
            for i in range(V):
                pi_feas_initial_shortage_U[i].setAttr(GRB.Attr.RHS, d_its[(i, 0, s)])

            # Update RHS for flow conservation constraints
            for i in range(V):
                for r in range(R):
                    for t in range(T - 1):
                        pi_feas_flow_conservation[i, r, t].setAttr(GRB.Attr.RHS, I_values[i, r, t + 1] - I_values[i, r, t])

            # Solve the feasibility subproblem
            sub_feas.optimize()

            # Store the objective value (should be positive if infeasible)
            w_feas[s] = sub_feas.objVal

            # Retrieve dual values from feasibility subproblem constraints
            for key, constraint in pi_feas_inventory_balance_U.items():
                pi_feas_inventory_balance_U_val[key] = constraint.pi
            for key, constraint in pi_feas_initial_shortage_U.items():
                pi_feas_initial_shortage_U_val[key] = constraint.pi
            for key, constraint in pi_feas_flow_conservation.items():
                pi_feas_flow_conservation_val[key] = constraint.pi
            for key, constraint in pi_feas_arc_capacity.items():
                pi_feas_arc_capacity_val[key] = constraint.pi

            # Generate feasibility cut if w_feas is large
            if w_feas[s] > eps:
                feasibility_cut_expr = gp.LinExpr()

                # Add inventory flow balance terms
                for key, dual_val in pi_feas_inventory_balance_U_val.items():
                    i, t = key
                    feasibility_cut_expr += dual_val * d_its[(i, t, s)]

                # Add initial shortage terms
                for key, dual_val in pi_feas_initial_shortage_U_val.items():
                    i = key
                    feasibility_cut_expr += dual_val * d_its[(i, 0, s)]

                # Add flow conservation terms
                for key, dual_val in pi_feas_flow_conservation_val.items():
                    i, r, t = key
                    if abs(dual_val) > 1e-7:
                        feasibility_cut_expr += dual_val * (I[i, r, t + 1] - I[i, r, t])

                # Add arc capacity terms
                for key, dual_val in pi_feas_arc_capacity_val.items():
                    i, j, t = key
                    feasibility_cut_expr += dual_val * -K_ijt[i][j][t]

                # Add the feasibility cut to the master problem
                master_model.addConstr(feasibility_cut_expr <= 0, name=f"feasibility_cut_scenario_{s}")
                Fcuts += 1  # Increment feasibility cuts counter

        else:
            print("here")
            # Compute a temporary upper bound using second-stage costs
            tempUB = sum(Q_opt[s] * P[s] for s in range(num))

            # Update the upper bound
            if tempUB < UB:
                UB = tempUB

            # Check convergence: If (UB - LB) is small, stop
            if (UB - LB) / max(1e-10, abs(LB)) < eps:
                break

            else:
                # Initialize the optimality cut expression
                optimality_cut_expr = gp.LinExpr()
                for s in range(num):
                    # Add inventory flow balance constraints
                    for key, dual_val in pi_opt_inventory_balance_U_val[s].items():
                        i, t = key
                        optimality_cut_expr += P[s] * dual_val * d_its[(i, t, s)]

                    # Add initial shortage constraints
                    for key, dual_val in pi_opt_initial_shortage_U_val[s].items():
                        i = key
                        optimality_cut_expr += P[s] * dual_val * d_its[(i, 0, s)]

                    # Add flow conservation constraints
                    for key, dual_val in pi_opt_flow_conservation_val[s].items():
                        i, r, t = key
                        if abs(dual_val) > 1e-7:
                            optimality_cut_expr += P[s] * dual_val * (I[i, r, t + 1] - I[i, r, t])

                    # Add arc capacity constraints
                    for key, dual_val in pi_opt_arc_capacity_val[s].items():
                        i, j, t = key
                        optimality_cut_expr += P[s] * dual_val * -K_ijt[i][j][t]

                # Add the optimality cut to the master model
                master_model.addConstr(theta >= optimality_cut_expr, name="optimality_cut")
                Ocuts += 1  # Increment the counter for optimality cuts

        # Print iteration details
        print(f"Iteration {iter}:")
        print(f"Lower Bound (LB) = {LB}")
        print(f"Upper Bound (UB) = {UB}")
        print("===========================================")

    # Final summary after convergence
    print("#########################################################################################")
    print(f"Final Iteration: {iter}")
    print(f"Number of Feasibility Cuts Added: {Fcuts}")
    print(f"Number of Optimality Cuts Added: {Ocuts}")
    print(f"Final Lower Bound (LB): {LB}")
    print(f"Final Upper Bound (UB): {UB}")
    print("#########################################################################################")

    # Return results
    return LB, UB, iter, Fcuts, Ocuts