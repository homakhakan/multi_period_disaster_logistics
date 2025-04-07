import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

def benders(model, eps, thetaLB, time_limit):

    V = model.network_processor.V
    R = model.network_processor.R
    T = model.network_processor.T
    A = model.network_processor.A
    V0 = model.network_processor.V0
    V1 = model.network_processor.V1
    V3 = model.network_processor.V3

    t_ijt = model.network_processor.t_ijt
    p_i = model.network_processor.p_i
    cost_ir = model.network_processor.cost_ir
    m_ij = model.network_processor.m_ij
    g_shelter = model.network_processor.g_shelter
    g_pod = model.network_processor.g_pod
    w_r = model.network_processor.w_r
    K_ijt = model.network_processor.K_ijt
    P = model.demand_processor.P
    sce = range(len(P))

    start_time = time.time()
     # Initialize bounds and tolerance
    model.LB = thetaLB
    model.UB = 1e10

    # Counters for optimality and feasibility cuts
    model.Ocuts = 0
    model.Fcuts = 0

    # Iteration counter
    model.iter = 0
    
    # Number of scenarios
    num = len(sce)

    # Store first-stage decision variable values
    model.I_values = {}
    model.theta_value = 0

    # Storage for optimality subproblem dual multipliers
    model.pi_opt_inventory_balance_U_val = {s: {} for s in range(num)}
    model.pi_opt_initial_shortage_U_val = {s: {} for s in range(num)}
    model.pi_opt_flow_conservation_val = {s: {} for s in range(num)}
    model.pi_opt_arc_capacity_val = {s: {} for s in range(num)}
    
    # Storage for feasibility subproblem dual multipliers
    model.pi_feas_inventory_balance_U_val = {}
    model.pi_feas_initial_shortage_U_val = {}
    model.pi_feas_flow_conservation_val = {}
    model.pi_feas_arc_capacity_val = {}

    # Start the Benders loop
    while (model.UB - model.LB) / max(1e-10, abs(model.UB)) > eps:
        model.iter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            print(f"\nStopping Benders after {elapsed_time:.2f} seconds (Time limit reached)")
            break  # Stop execution if the time limit is reached

        # Solve the master problem
        model.master_model.optimize()

        # Check if the optimization was successful
        if model.master_model.status == GRB.OPTIMAL:
            # Extract the values of first-stage decision variables I[i, r, t]
            for i in range(V):
                for r in range(R):
                    for t in range(T):
                        model.I_values[i, r, t] = model.I[i, r, t].x  # Store decision variable values

            # Get the value of theta
            model.theta_value = model.theta.x

            # Update the lower bound
            model.LB = model.master_model.objVal       
            
        elif model.master_model.status == GRB.INFEASIBLE:
            print("Master problem is infeasible.")
            exit(0)  # Or handle infeasibility appropriately

        elif model.master_model.status == GRB.UNBOUNDED:
            print("Master problem is unbounded.")
            exit(0)  # Or handle unboundedness appropriately

        else:
            print(f"Optimization was stopped with status {model.master_model.status}")
            exit(0)

        # Initialize storage for subproblem solutions
        model.Q_opt = np.zeros(num)  # Stores optimality subproblem objective values
        model.w_feas = np.zeros(num)  # Stores feasibility subproblem objective values
        model.FLAG = 0  # (0 if I_hat is feasible for all scenarios, 1 otherwise)
        model.infeasible_sce = -1  # Track which scenario is infeasible

        # Loop over all scenarios
        for s in range(num):
            # Update RHS for inventory flow balance constraints
            for i in range(V):
                for t in range(1, T):
                    model.pi_opt_inventory_balance_U[i, t].setAttr(GRB.Attr.RHS, model.demand_processor.d_its[(i, t, s)])

            # Update RHS for initial shortage constraints
            for i in range(V):
                model.pi_opt_initial_shortage_U[i].setAttr(GRB.Attr.RHS, model.demand_processor.d_its[(i, 0, s)])

            # Update RHS for flow conservation constraints
            for i in range(V):
                for r in range(R):
                    for t in range(T - 1):
                        model.pi_opt_flow_conservation[i, r, t].setAttr(GRB.Attr.RHS, model.I_values[i, r, t + 1] - model.I_values[i, r, t])

            # Solve the optimality subproblem
            model.sub_opt.optimize()

            if model.sub_opt.status == GRB.OPTIMAL:
                # Store the optimal objective value
                model.Q_opt[s] = model.sub_opt.objVal

                # Retrieve and store dual values for each constraint
                for key, constraint in model.pi_opt_inventory_balance_U.items():
                    model.pi_opt_inventory_balance_U_val[s][key] = constraint.pi
                for key, constraint in model.pi_opt_initial_shortage_U.items():
                    model.pi_opt_initial_shortage_U_val[s][key] = constraint.pi
                for key, constraint in model.pi_opt_flow_conservation.items():
                    model.pi_opt_flow_conservation_val[s][key] = constraint.pi
                for key, constraint in model.pi_opt_arc_capacity.items():
                    model.pi_opt_arc_capacity_val[s][key] = constraint.pi

            elif model.sub_opt.status == GRB.INFEASIBLE:
                # If infeasible, set FLAG to 1 and mark the scenario
                model.FLAG = 1
                model.infeasible_sce = s
                break;  # No need to continue checking other scenarios

            else:
                print(f"Error: One of the scenario subproblems is infeasible (Status: {model.sub_opt.status})")
                exit(0)

        print("FLAG =", model.FLAG)

        if model.FLAG == 1:
            # Solve feasibility subproblem for the detected infeasible scenario
            s = model.infeasible_sce

            # Update RHS for inventory flow balance constraints
            for i in range(V):
                for t in range(1, T):
                    model.pi_feas_inventory_balance_U[i, t].setAttr(GRB.Attr.RHS, model.demand_processor.d_its[(i, t, s)])

            # Update RHS for initial shortage constraints
            for i in range(V):
                model.pi_feas_initial_shortage_U[i].setAttr(GRB.Attr.RHS, model.demand_processor.d_its[(i, 0, s)])

            # Update RHS for flow conservation constraints
            for i in range(V):
                for r in range(R):
                    for t in range(T - 1):
                        model.pi_feas_flow_conservation[i, r, t].setAttr(GRB.Attr.RHS, model.I_values[i, r, t + 1] - model.I_values[i, r, t])

            # Solve the feasibility subproblem
            model.sub_feas.optimize()

            # Store the objective value (should be positive if infeasible)
            model.w_feas[s] = model.sub_feas.objVal

            # Retrieve dual values from feasibility subproblem constraints
            for key, constraint in model.pi_feas_inventory_balance_U.items():
                model.pi_feas_inventory_balance_U_val[key] = constraint.pi
            for key, constraint in model.pi_feas_initial_shortage_U.items():
                model.pi_feas_initial_shortage_U_val[key] = constraint.pi
            for key, constraint in model.pi_feas_flow_conservation.items():
                model.pi_feas_flow_conservation_val[key] = constraint.pi
            for key, constraint in model.pi_feas_arc_capacity.items():
                model.pi_feas_arc_capacity_val[key] = constraint.pi

            # Generate feasibility cut if w_feas is large
            if model.w_feas[s] > eps:
                feasibility_cut_expr = gp.LinExpr()

                # Add inventory flow balance terms
                for key, dual_val in model.pi_feas_inventory_balance_U_val.items():
                    i, t = key
                    feasibility_cut_expr += dual_val * model.demand_processor.d_its[(i, t, s)]

                # Add initial shortage terms
                for key, dual_val in model.pi_feas_initial_shortage_U_val.items():
                    i = key
                    feasibility_cut_expr += dual_val * model.demand_processor.d_its[(i, 0, s)]

                # Add flow conservation terms
                for key, dual_val in model.pi_feas_flow_conservation_val.items():
                    i, r, t = key
                    if abs(dual_val) > 1e-7:
                        feasibility_cut_expr += dual_val * (model.I[i, r, t + 1] - model.I[i, r, t])

                # Add arc capacity terms
                for key, dual_val in model.pi_feas_arc_capacity_val.items():
                    i, j, t = key
                    feasibility_cut_expr += dual_val * -K_ijt[i][j][t]

                # Add the feasibility cut to the master problem
                model.master_model.addConstr(feasibility_cut_expr <= 0, name=f"feasibility_cut_scenario_{s}")
                model.Fcuts += 1  # Increment feasibility cuts counter

        else:
            print("here")
            # Compute a temporary upper bound using second-stage costs
            model.tempUB = sum(model.Q_opt[s] * P[s] for s in range(num))

            # Update the upper bound
            if model.tempUB < model.UB:
                model.UB = model.tempUB

            # Check convergence: If (UB - LB) is small, stop
            if (model.UB - model.LB) / max(1e-10, abs(model.LB)) < eps:
                break

            else:
                # Initialize the optimality cut expression
                optimality_cut_expr = gp.LinExpr()
                for s in range(num):
                    # Add inventory flow balance constraints
                    for key, dual_val in model.pi_opt_inventory_balance_U_val[s].items():
                        i, t = key
                        optimality_cut_expr += P[s] * dual_val * model.demand_processor.d_its[(i, t, s)]

                    # Add initial shortage constraints
                    for key, dual_val in model.pi_opt_initial_shortage_U_val[s].items():
                        i = key
                        optimality_cut_expr += P[s] * dual_val * model.demand_processor.d_its[(i, 0, s)]

                    # Add flow conservation constraints
                    for key, dual_val in model.pi_opt_flow_conservation_val[s].items():
                        i, r, t = key
                        if abs(dual_val) > 1e-7:
                            optimality_cut_expr += P[s] * dual_val * (model.I[i, r, t + 1] - model.I[i, r, t])

                    # Add arc capacity constraints
                    for key, dual_val in model.pi_opt_arc_capacity_val[s].items():
                        i, j, t = key
                        optimality_cut_expr += P[s] * dual_val * -K_ijt[i][j][t]

                # Add the optimality cut to the master model
                model.master_model.addConstr(model.theta >= optimality_cut_expr, name="optimality_cut")
                model.Ocuts += 1  # Increment the counter for optimality cuts

        # Print iteration details
        print(f"Iteration {model.iter}:")
        print(f"Lower Bound (LB) = {model.LB}")
        print(f"Upper Bound (UB) = {model.UB}")
        print("===========================================")

    # Final summary after convergence
    print("#########################################################################################")
    print(f"Final Iteration: {model.iter}")
    print(f"Number of Feasibility Cuts Added: {model.Fcuts}")
    print(f"Number of Optimality Cuts Added: {model.Ocuts}")
    print(f"Final Lower Bound (LB): {model.LB}")
    print(f"Final Upper Bound (UB): {model.UB}")
    print("#########################################################################################")

    # Return results
    return model.LB, model.UB, model.iter, model.Fcuts, model.Ocuts

