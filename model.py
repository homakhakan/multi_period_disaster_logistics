import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

### Extended
def create_ext(sets, parameters, d_its, num_sce):

    # 🔹 Initialize the Gurobi Model
    m = gp.Model("Two-Stage Stochastic Programming Model")

    # 🔹 Define Decision Variables
    I, U, S, f = dvs_ext(m, sets, d_its, num_sce)

    # 🔹 Define Constraints
    cons_ext(m, sets, parameters, d_its, I, U, S, f, num_sce)

    # 🔹 Define Objective Function
    obj_ext(m, sets, parameters, I, U, S, f)

    return m  # Returns the complete model

def dvs_ext(m, sets, d_its, num_sce):

    V, R, T, A = sets["V"], sets["R"], sets["T"], sets["A"]
    sce = range(num_sce)  # Number of scenarios

    # I: Inventory of commodity r at node i at the start of time period t
    I = {}
    
    for i in range(V):
        for r in range(R):
            for t in range(T):
                I[i, r, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="I_%s_%s_%s" % (i, r, t))
    
    # Define second-stage decision variables
    U = {}  # Demand shortage at location i at time period t in scenario s
    
    for i in range(V):
        for t in range(T):
            for s in sce:  # Iterate directly over the scenario set (e.g., 0 and 1)
                U[i, t, s] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="U_%s_%s_%s" % (i, t, s))
    
    S = {}  # Demand satisfied at location i at time period t in scenario s
    
    for i in range(V):
        for t in range(T):
            for s in sce:  # Iterate directly over the scenario set
                S[i, t, s] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S_%s_%s_%s" % (i, t, s))
    
    f = {}  # Quantity of commodity r traveling on link (i, j) at time period t in scenario s
    
    for i in range(V):
        for j in range(V):
            if (i, j) in A:  # Only define f_ijrt^s if (i, j) is a valid link
                for r in range(R):
                    for t in range(T):
                        for s in sce:  # Iterate directly over the scenario set
                            f[i, j, r, t, s] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_%s_%s_%s_%s_%s" % (i, j, r, t, s))
    m.update()
    return I, U, S, f

def cons_ext(m, sets, parameters, d_its, I, U, S, f, num_sce):

    V, V0, V1, V3, R, T, T_Bef, A = sets["V"], sets["V0"], sets["V1"], sets["V3"], sets["R"], sets["T"], sets["T_Bef"], sets["A"]
    K_ijt, w_r, I_ir0, g_shelter, g_pod, t_ijt = parameters["K_ijt"], parameters["w_r"], parameters["I_ir0"], parameters["g_shelter"], parameters["g_pod"], parameters["t_ijt"]
    sce = range(num_sce)  # Number of scenarios

    # Add constraints for the lower bound of inventory at warehouses (V0)
    for i in V0:  # Loop over warehouse nodes
        for r in range(R):  # Loop over commodities
            for t in range(1, T):  # Loop over time periods (from t=1 to T-1)
                # Calculate the total arc capacity for the current time period
                arc_capacity = sum(K_ijt[i][j][t] for j in range(V) if (i, j) in A)
                # Calculate max outflow for commodity r
                max_outflow_r = arc_capacity  / w_r[r]
                # Check if the initial inventory is greater than or equal to max_outflow_r
                if I_ir0[i][r] >= max_outflow_r:
                    # Add the lower bound constraint
                    m.addConstr(I[i, r, t] >= I_ir0[i][r] - max_outflow_r, name=f"LowerBound_Warehouse_{i}_{r}_{t}")
    
    
    # Add constraints for the upper bound of inventory at PoDs (V3)
    for i in V3:  # PoDs
        for r in range(R):  # Commodities
            for t in T_Bef[1:]:  # Time periods before landfall, excluding t=0
                # Calculate the total arc capacity for the current time period
                arc_capacity = sum(K_ijt[j][i][t] for j in range(V) if (j, i) in A)
                # Calculate max inflow for commodity r
                max_inflow_r = arc_capacity / w_r[r]            
                # Add the upper bound constraint
                m.addConstr(I[i, r, t] <= I_ir0[i][r] + max_inflow_r, name=f"UpperBound_PoD_{i}_{r}_{t}")
    
    
    # Add constraints to ensure inventory at warehouses only decreases over time
    for i in V0:
        for r in range(R):
            for t in range(1, T):
                m.addConstr(
                    I[i, r, t] <= I[i, r, t-1],
                    name=f"inventory_decrease_warehouse_{i}_{r}_{t}")
    
    # Add constraints to ensure inventory at PoDs only increases before landfall
    for i in V3:
        for r in range(R):
            for t in T_Bef[1:]:
                m.addConstr(
                    I[i, r, t] >= I[i, r, t-1],
                    name=f"inventory_increase_pod_{i}_{r}_{t}")
    
    m.update()  # Ensure the model is updated with the latest constraints
    #m.write("model_constraints.lp")
    
    #Starting Value of I[t=0]
    for i in range(V):
        for r in range(R):
            m.addConstr(I[i,r,0]==I_ir0[i][r])
    
    # Update demand shortage for all locations, time periods (except t=0), and scenarios
    for i in range(V):
        for t in range(1, T):
            for s in sce:
                m.addConstr(U[i, t, s] == U[i, t-1, s] - S[i, t, s] + d_its[(i, t, s)], 
                            name="DemandShortageUpdate_%s_%s_%s" % (i, t, s))
    
    # Initial demand shortage for all locations and scenarios at time period t=0
    for i in range(V):
        for s in sce:
            m.addConstr(U[i, 0, s] == d_its[(i, 0, s)], name="InitialDemandShortage_%s_%s" % (i, s))
    
    # Inventory update constraint for all locations, commodities, time periods (except last), and scenarios
    for i in range(V):
        for r in range(R):
            for t in range(T - 1):
                for s in sce:
                    outgoing_flow = gp.quicksum(f[i, j, r, t, s] for j in range(V) if (i, j) in A)
                    incoming_flow = gp.quicksum(f[j, i, r, t - t_ijt[j][i] + 1, s] for j in range(V) if (j, i) in A and t - t_ijt[j][i] + 1 >= 0)
                    
                    # Inventory update constraint for shelters
                    if i in V1:
                        m.addConstr(
                            I[i, r, t + 1] == I[i, r, t] - outgoing_flow + incoming_flow - S[i, t, s] * g_shelter[r],
                            name="InventoryUpdate_Shelter_%s_%s_%s_%s" % (i, r, t, s)
                        )
                    # Inventory update constraint for pods
                    elif i in V3:
                        m.addConstr(
                            I[i, r, t + 1] == I[i, r, t] - outgoing_flow + incoming_flow - S[i, t, s] * g_pod[r],
                            name="InventoryUpdate_Pod_%s_%s_%s_%s" % (i, r, t, s)
                        )
                    # Inventory update constraint for warehouses and RSAs
                    else:
                        m.addConstr(
                            I[i, r, t + 1] == I[i, r, t] - outgoing_flow + incoming_flow,
                            name="InventoryUpdate_%s_%s_%s_%s" % (i, r, t, s)
                        )
    
    
    # Flow capacity constraint for all links, time periods, and scenarios
    for i in range(V):
        for j in range(V):
            if (i, j) in A:  # Only consider valid links in A
                for t in range(T):
                    for s in sce:
                        # Check if the capacity for this (i, j, t) exists in K_ijt
                        #if i in K_ijt and j in K_ijt[i] and t in K_ijt[i][j]:
                            m.addConstr(
                                gp.quicksum(w_r[r] * f[i, j, r, t, s] for r in range(R)) <= K_ijt[i][j][t],
                                name="FlowCapacity_%s_%s_%s_%s" % (i, j, t, s)
                            )
    
    m.update()

def obj_ext(m, sets, parameters, I, U, S, f):

    P, p_i, cost_ir, m_ij = parameters["P"], parameters["p_i"], parameters["cost_ir"], parameters["m_ij"]
    sce = range(len(P))
    # Convert T and R to ranges before using them in loops
    T_range = range(sets["T"]) if isinstance(sets["T"], int) else sets["T"]
    R_range = range(sets["R"]) if isinstance(sets["R"], int) else sets["R"]
    V_range = range(sets["V"]) if isinstance(sets["V"], int) else sets["V"]

    objective = gp.quicksum(
        P[s] * (
            gp.quicksum(
                gp.quicksum(p_i[i] * U[i, t, s] for i in V_range) for t in T_range
            ) +
            gp.quicksum(
                gp.quicksum(
                    gp.quicksum(cost_ir[i][r] * f[i, j, r, t, s] for i in sets["V0"] for j in V_range if (i, j) in sets["A"]) +
                    gp.quicksum(m_ij[i][j] * f[i, j, r, t, s] for i in V_range for j in V_range if (i, j) in sets["A"])
                    for r in R_range
                ) for t in T_range
            )
        ) for s in sce
    )
        
    m.setObjective(objective, GRB.MINIMIZE)
    m.update()

### Benders
def master_problem(sets, parameters, thetaLB):
    # Extract required sets and parameters
    V, R, T, V0, V3, T_Bef, A = sets["V"], sets["R"], sets["T"], sets["V0"], sets["V3"], sets["T_Bef"], sets["A"]
    I_ir0, K_ijt, w_r = parameters["I_ir0"], parameters["K_ijt"], parameters["w_r"]

    # Create master model
    master_model = gp.Model("MasterProblem")
    master_model.modelSense = GRB.MINIMIZE

    #Define first-stage decision variables
    I = {}
    for i in range(V):
        for r in range(R):
            for t in range(T):
                I[i, r, t] = master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{r}_{t}")

    #Define the auxiliary variable
    theta = master_model.addVar(vtype=GRB.CONTINUOUS, lb=thetaLB, name="theta")

    #Add constraints
    #Initial inventory levels
    for i in range(V):
        for r in range(R):
            master_model.addConstr(I[i, r, 0] == I_ir0[i][r], name=f"initial_inventory_{i}_{r}")

    #Ensure inventory at warehouses only decreases over time
    for i in V0:
        for r in range(R):
            for t in range(1, T):
                master_model.addConstr(I[i, r, t] <= I[i, r, t-1], name=f"inventory_decrease_warehouse_{i}_{r}_{t}")

    #Ensure inventory at PoDs only increases before landfall
    for i in V3:
        for r in range(R):
            for t in T_Bef[1:]:
                master_model.addConstr(I[i, r, t] >= I[i, r, t-1], name=f"inventory_increase_pod_{i}_{r}_{t}")

    #Lower bound of inventory at warehouses (V0)
    for i in V0:
        for r in range(R):
            for t in range(1, T): 
                # Calculate the total arc capacity for the current time period
                arc_capacity = sum(K_ijt[i][j][t] for j in range(V) if (i, j) in A)
                # Calculate max outflow for commodity r
                max_outflow_r = arc_capacity  / w_r[r]
                if I_ir0[i][r] >= max_outflow_r:
                    master_model.addConstr(I[i, r, t] >= I_ir0[i][r] - max_outflow_r, name=f"LowerBound_Warehouse_{i}_{r}_{t}")

    #Upper bound of inventory at PoDs (V3)
    for i in V3:
        for r in range(R):
            for t in T_Bef[1:]:
                # Calculate the total arc capacity for the current time period
                arc_capacity = sum(K_ijt[j][i][t] for j in range(V) if (j, i) in A)
                # Calculate max inflow for commodity r
                max_inflow_r = arc_capacity / w_r[r]            
                master_model.addConstr(I[i, r, t] <= I_ir0[i][r] + max_inflow_r, name=f"UpperBound_PoD_{i}_{r}_{t}")

    master_model.setObjective(theta, GRB.MINIMIZE)
    master_model.setParam("OutputFlag", 0)
    master_model.update()
    
    return master_model, I, theta

def optimality_sub(sets, parameters):
    # Extract required sets and parameters
    V, R, T, A, V0, V1, V3 = sets["V"], sets["R"], sets["T"], sets["A"], sets["V0"], sets["V1"], sets["V3"]
    t_ijt, p_i, cost_ir, m_ij, g_shelter, g_pod, w_r, K_ijt = (
        parameters["t_ijt"], parameters["p_i"], parameters["cost_ir"], parameters["m_ij"],
        parameters["g_shelter"], parameters["g_pod"], parameters["w_r"], parameters["K_ijt"]
    )

    # Create optimality subproblem model
    sub_opt = gp.Model("OptimalitySubproblem")
    sub_opt.modelSense = GRB.MINIMIZE

    #Define decision variables
    U_opt = {}
    S_opt = {}
    f_opt = {}
    
    # U: Demand shortage at location i at time period t in scenario s
    for i in range(V):
        for t in range(T):
            U_opt[i, t] = sub_opt.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_opt_{i}_{t}")
    
    # S: Demand satisfied at location i at time period t in scenario s
    for i in range(V):
        for t in range(T):
            S_opt[i, t] = sub_opt.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_opt_{i}_{t}")
    
    # f: Quantity of commodity r traveling on link (i, j) at time period t in scenario s
    for i in range(V):
        for j in range(V):
            if (i, j) in A:
                for r in range(R):
                    for t in range(T):
                        f_opt[i, j, r, t] = sub_opt.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_opt_{i}_{j}_{r}_{t}")

    #Objective function
    sub_opt.setObjective(
        gp.quicksum(  # Demand shortage cost
            gp.quicksum(p_i[i] * U_opt[i, t] for i in range(V))
            for t in range(T)) +
        gp.quicksum(  # Procurement and transportation costs
            gp.quicksum(
                gp.quicksum(
                    cost_ir[i][r] * f_opt[i, j, r, t]  # Procurement cost at V0
                    for i in V0 for j in range(V) if (i, j) in A) +
                gp.quicksum(
                    m_ij[i][j] * f_opt[i, j, r, t]  # Transportation cost for all locations
                    for i in range(V) for j in range(V) if (i, j) in A) 
                for r in range(R)) for t in range(T)))

    #Constraints
    # Constraint for inventory flow balance with U and S
    pi_opt_inventory_balance_U = {}
    for i in range(V):
        for t in range(1, T):  # Start from t=1 since t=0 is the initial condition
            pi_opt_inventory_balance_U[i, t] = sub_opt.addConstr(U_opt[i, t] - U_opt[i, t-1] + S_opt[i, t] == 0, 
                                                                 name=f"inventory_balance_U_opt_{i}_{t}")
    
    # Initial shortage for demand at t=0
    pi_opt_initial_shortage_U = {}
    for i in range(V):
        pi_opt_initial_shortage_U[i] = sub_opt.addConstr(U_opt[i, 0] == 0, name=f"initial_shortage_U_opt_{i}")
    
    # Flow conservation constraint for f with the RHS set to 0 temporarily, using appropriate g_shelter or g_pod
    pi_opt_flow_conservation = {}
    for i in range(V):
        for r in range(R):
            for t in range(T - 1):
                # Add the constraint to the model
                pi_opt_flow_conservation[i, r, t] = sub_opt.addConstr(
                    -gp.quicksum(f_opt[i, j, r, t] for j in range(V) if (i, j) in A)
                    + gp.quicksum(f_opt[j, i, r, t - t_ijt[j][i] + 1] for j in range(V) if (i, j) in A and t - t_ijt[j][i] + 1 >= 0)
                    - S_opt[i, t] * (g_shelter[r] if i in V1 else g_pod[r] if i in V3 else 0) == 0,
                    name=f"flow_conservation_opt_{i}_{r}_{t}")
    
    # Arc capacity constraint
    pi_opt_arc_capacity = {}
    for i in range(V):
        for j in range(V):
            if (i, j) in A:  # Only consider valid links in A
                for t in range(T):
                        pi_opt_arc_capacity[i, j, t] = sub_opt.addConstr(-gp.quicksum(w_r[r] * f_opt[i, j, r, t] for r in range(R)) >= -K_ijt[i][j][t],
                            name=f"arc_capacity_opt_{i}_{j}_{t}")

    sub_opt.setParam("OutputFlag", 0)

    return sub_opt, U_opt, S_opt, f_opt, pi_opt_inventory_balance_U, pi_opt_initial_shortage_U, pi_opt_flow_conservation, pi_opt_arc_capacity

def feasibility_sub(sets, parameters):
    # Extract required sets and parameters
    V, R, T, A, V1, V3 = sets["V"], sets["R"], sets["T"], sets["A"], sets["V1"], sets["V3"]
    t_ijt, g_shelter, g_pod, w_r, K_ijt = (
        parameters["t_ijt"], parameters["g_shelter"], parameters["g_pod"], parameters["w_r"], parameters["K_ijt"]
    )

    # Create feasibility subproblem model
    sub_feas = gp.Model("FeasibilitySubproblem")
    sub_feas.modelSense = GRB.MINIMIZE

    #Define decision variables
    U_feas = {}
    S_feas = {}
    f_feas = {}
    
    # U: Demand shortage at location i at time period t in scenario s
    for i in range(V):
        for t in range(T):
            U_feas[i, t] = sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f"U_feas_{i}_{t}")
    
    # S: Demand satisfied at location i at time period t in scenario s
    for i in range(V):
        for t in range(T):
            S_feas[i, t] = sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f"S_feas_{i}_{t}")
    
    # f: Quantity of commodity r traveling on link (i, j) at time period t in scenario s
    for i in range(V):
        for j in range(V):
            if (i, j) in A:
                for r in range(R):
                    for t in range(T):
                        f_feas[i, j, r, t] = sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f"f_feas_{i}_{j}_{r}_{t}")

    #Define slack variables ε (epsilon) for each group in the objective function
    epsilon_1 = {}
    for i in range(V):
        for r in range(R):
            for t in range(T-1):
                epsilon_1[i, r, t] = sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"epsilon1_{i}_{r}_{t}")
    
    epsilon_2 = {}
    for i in range(V):
        for r in range(R):
            for t in range(T-1):
                epsilon_2[i, r, t] = sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"epsilon2_{i}_{r}_{t}")
    
    epsilon_3 = {}
    for i in range(V):
        for j in range(V):
            if (i, j) in A:
                for t in range(T):
                    epsilon_3[i, j, t] = sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"epsilon3_{i}_{j}_{t}")

    sub_feas.setObjective(
    gp.quicksum(epsilon_1[i, r, t] + epsilon_2[i, r, t] for i in range(V) for r in range(R) for t in range(T - 1)) +
    gp.quicksum(epsilon_3[i, j, t] for i in range(V) for j in range(V) if (i, j) in A for t in range(T)))

    # Constraints
    # Constraint: Inventory flow balance for U, S, and epsilon variables
    pi_feas_inventory_balance_U = {}
    for i in range(V):
        for t in range(1, T):  # Note: starting from t=1 as per the constraint
            pi_feas_inventory_balance_U[i, t] = sub_feas.addConstr(
                U_feas[i, t] - U_feas[i, t-1] + S_feas[i, t] == 0,
                name=f"inventory_balance_U_feas_{i}_{t}")
    
    # Initialize shortage for U at t=0
    pi_feas_initial_shortage_U = {}
    for i in range(V):
        pi_feas_initial_shortage_U[i] = sub_feas.addConstr(U_feas[i, 0] == 0, name=f"initial_shortage_U_feas_{i}")
    
    # Flow conservation constraint
    pi_feas_flow_conservation = {}
    for i in range(V):
        for r in range(R):
            for t in range(T - 1):
                pi_feas_flow_conservation[i, r, t] = sub_feas.addConstr(
                    -gp.quicksum(f_feas[i, j, r, t] for j in range(V) if (i, j) in A)
                    + gp.quicksum(f_feas[j, i, r, t - t_ijt[j][i] + 1] for j in range(V) if (i, j) in A and t - t_ijt[j][i] + 1 >= 0)
                    - S_feas[i, t] * (g_shelter[r] if i in V1 else g_pod[r] if i in V3 else 0)
                    + epsilon_1[i, r, t] - epsilon_2[i, r, t] == 0,
                    name=f"flow_conservation_feas_{i}_{r}_{t}")
    
    # Flow capacity constraint for all links, time periods, and scenarios
    pi_feas_arc_capacity = {}
    for i in range(V):
        for j in range(V):
            if (i, j) in A:  # Only consider valid links in A
                for t in range(T):
                    pi_feas_arc_capacity[i, j, t] = sub_feas.addConstr(
                        -gp.quicksum(w_r[r] * f_feas[i, j, r, t] for r in range(R))
                        + epsilon_3[i, j, t] >= -K_ijt[i][j][t],
                        name=f"arc_capacity_feas_{i}_{j}_{t}")

    # Disable Gurobi solver output
    sub_feas.setParam("OutputFlag", 0)

    return sub_feas, U_feas, S_feas, f_feas, epsilon_1, epsilon_2, epsilon_3, pi_feas_inventory_balance_U, pi_feas_initial_shortage_U, pi_feas_flow_conservation, pi_feas_arc_capacity

### Mean Value Problem
def create_mvp():
    return gp.Model("Mean Value Problem")

def dvs_mvp(mvp, sets):
    V, R, T, A = sets["V"], sets["R"], sets["T"], sets["A"]

    # Define first-stage decision variables
    # I: Inventory of commodity r at node i at the start of time period t
    I_mvp = {}
    
    for i in range(V):
        for r in range(R):
            for t in range(T):
                I_mvp[i, r, t] = mvp.addVar(vtype=GRB.CONTINUOUS, name="I_mvp_%s_%s_%s" % (i, r, t))
    
    # Define second-stage decision variables
    U_mvp = {}  # Demand shortage at location i at time period t in scenario s
    
    for i in range(V):
        for t in range(T):
            U_mvp[i, t] = mvp.addVar(vtype=GRB.CONTINUOUS, name="U_mvp_%s_%s" % (i, t))
    
    S_mvp = {}  # Demand satisfied at location i at time period t in scenario s
    
    for i in range(V):
        for t in range(T):
            S_mvp[i, t] = mvp.addVar(vtype=GRB.CONTINUOUS, name="S_mvp_%s_%s" % (i, t))
    
    f_mvp = {}  # Quantity of commodity r traveling on link (i, j) at time period t in scenario s
    
    for i in range(V):
        for j in range(V):
            if (i, j) in A:  # Only define f_ijrt^s if (i, j) is a valid link
                for r in range(R):
                    for t in range(T):
                        f_mvp[i, j, r, t] = mvp.addVar(vtype=GRB.CONTINUOUS, name="f_mvp_%s_%s_%s_%s" % (i, j, r, t))
    mvp.update()

    return I_mvp, U_mvp, S_mvp, f_mvp
    
def cons_mvp(mvp, I_mvp, U_mvp, S_mvp, f_mvp, sets, parameters, d_it):

    V, V0, V1, V3, R, T, T_Bef, A = sets["V"], sets["V0"], sets["V1"], sets["V3"], sets["R"], sets["T"], sets["T_Bef"], sets["A"]
    K_ijt, w_r, I_ir0, g_shelter, g_pod, t_ijt = parameters["K_ijt"], parameters["w_r"], parameters["I_ir0"], parameters["g_shelter"], parameters["g_pod"], parameters["t_ijt"]

    # Add constraints for the lower bound of inventory at warehouses (V0)
    for i in V0:  # Loop over warehouse nodes
        for r in range(R):  # Loop over commodities
            for t in range(1, T):  # Loop over time periods (from t=1 to T-1)
                # Calculate the total arc capacity for the current time period
                arc_capacity = sum(K_ijt[i][j][t] for j in range(V) if (i, j) in A)
                # Calculate max outflow for commodity r
                max_outflow_r = arc_capacity  / w_r[r]
                # Check if the initial inventory is greater than or equal to max_outflow_r
                if I_ir0[i][r] >= max_outflow_r:
                    # Add the lower bound constraint
                    mvp.addConstr(I_mvp[i, r, t] >= I_ir0[i][r] - max_outflow_r, name=f"LowerBound_Warehouse_mvp_{i}_{r}_{t}")
    
    # Add constraints for the upper bound of inventory at PoDs (V3)
    for i in V3:  # PoDs
        for r in range(R):  # Commodities
            for t in T_Bef[1:]:  # Time periods before landfall, excluding t=0
                # Calculate the total arc capacity for the current time period
                arc_capacity = sum(K_ijt[j][i][t] for j in range(V) if (j, i) in A)
                # Calculate max inflow for commodity r
                max_inflow_r = arc_capacity / w_r[r]            
                # Add the upper bound constraint
                mvp.addConstr(I_mvp[i, r, t] <= I_ir0[i][r] + max_inflow_r, name=f"UpperBound_PoD_mvp_{i}_{r}_{t}")
       
    # Add constraints to ensure inventory at warehouses only decreases over time
    for i in V0:
        for r in range(R):
            for t in range(1, T):
                mvp.addConstr(
                    I_mvp[i, r, t] <= I_mvp[i, r, t-1],
                    name=f"inventory_decrease_warehouse_mvp_{i}_{r}_{t}")
    
    # Add constraints to ensure inventory at PoDs only increases before landfall
    for i in V3:
        for r in range(R):
            for t in T_Bef[1:]:
                mvp.addConstr(
                    I_mvp[i, r, t] >= I_mvp[i, r, t-1],
                    name=f"inventory_increase_pod_mvp_{i}_{r}_{t}")
    
    #Starting Value of I[t=0]
    for i in range(V):
        for r in range(R):
            mvp.addConstr(I_mvp[i,r,0]==I_ir0[i][r])
    
    # Update demand shortage for all locations, time periods (except t=0), and scenarios
    for i in range(V):
        for t in range(1, T):
            mvp.addConstr(U_mvp[i, t] == U_mvp[i, t-1] - S_mvp[i, t] + d_it[(i, t)], 
                            name="DemandShortageUpdate_mvp_%s_%s" % (i, t))
    
    # Initial demand shortage for all locations and scenarios at time period t=0
    for i in range(V):
        mvp.addConstr(U_mvp[i, 0] == d_it[(i, 0)], name="InitialDemandShortage_mvp_%s" % (i))
    
    # Inventory update constraint for all locations, commodities, time periods (except last), and scenarios
    for i in range(V):
        for r in range(R):
            for t in range(T - 1):
                outgoing_flow = gp.quicksum(f_mvp[i, j, r, t] for j in range(V) if (i, j) in A)
                incoming_flow = gp.quicksum(f_mvp[j, i, r, t - t_ijt[j][i] + 1] for j in range(V) if (j, i) in A and t - t_ijt[j][i] + 1 >= 0)
                    
                # Inventory update constraint for shelters
                if i in V1:
                    mvp.addConstr(
                        I_mvp[i, r, t + 1] == I_mvp[i, r, t] - outgoing_flow + incoming_flow - S_mvp[i, t] * g_shelter[r],
                        name="InventoryUpdate_Shelter_mvp_%s_%s_%s" % (i, r, t)
                    )
                # Inventory update constraint for pods
                elif i in V3:
                    mvp.addConstr(
                        I_mvp[i, r, t + 1] == I_mvp[i, r, t] - outgoing_flow + incoming_flow - S_mvp[i, t] * g_pod[r],
                        name="InventoryUpdate_Pod_mvp_%s_%s_%s" % (i, r, t)
                    )
                # Inventory update constraint for warehouses and RSAs
                else:
                    mvp.addConstr(
                        I_mvp[i, r, t + 1] == I_mvp[i, r, t] - outgoing_flow + incoming_flow,
                        name="InventoryUpdate_mvp_%s_%s_%s" % (i, r, t)
                    )
    
    
    # Flow capacity constraint for all links, time periods, and scenarios
    for i in range(V):
        for j in range(V):
            if (i, j) in A:  # Only consider valid links in A
                for t in range(T):
                    # Check if the capacity for this (i, j, t) exists in K_ijt
                    #if i in K_ijt and j in K_ijt[i] and t in K_ijt[i][j]:
                        mvp.addConstr(
                            gp.quicksum(w_r[r] * f_mvp[i, j, r, t] for r in range(R)) <= K_ijt[i][j][t],
                            name="FlowCapacity_mvp_%s_%s_%s" % (i, j, t)
                        )
    
    mvp.update()

def obj_mvp(mvp, I_mvp, U_mvp, S_mvp, f_mvp, sets, parameters):

    V, V0, R, T, A = sets["V"], sets["V0"], sets["R"], sets["T"], sets["A"]
    p_i, cost_ir, m_ij = parameters["p_i"], parameters["cost_ir"], parameters["m_ij"]
    # Convert T and R to ranges before using them in loops
    T_range = range(sets["T"]) if isinstance(sets["T"], int) else sets["T"]
    R_range = range(sets["R"]) if isinstance(sets["R"], int) else sets["R"]
    V_range = range(sets["V"]) if isinstance(sets["V"], int) else sets["V"]

    objective = gp.quicksum(
        gp.quicksum(p_i[i] * U_mvp[i, t] for i in V_range) for t in T_range
    ) + gp.quicksum(
        gp.quicksum(
            gp.quicksum(cost_ir[i][r] * f_mvp[i, j, r, t] for i in V0 for j in V_range if (i, j) in A) +
            gp.quicksum(m_ij[i][j] * f_mvp[i, j, r, t] for i in V_range for j in V_range if (i, j) in A)
            for r in R_range
        ) for t in T_range
    )
    
    mvp.setObjective(objective, GRB.MINIMIZE)
    mvp.update()

def compute_mvp_objective(sets, parameters, d_its, I_mvp_values, sub_opt, pi_opt_inventory_balance_U, 
                          pi_opt_initial_shortage_U, pi_opt_flow_conservation):
    
    
    P, V, R, T = parameters["P"], sets["V"], sets["R"], sets["T"]
    sce = range(len(P))
    mvpobj = 0  # Initialize the MVP objective value

    # Loop over all scenarios to update RHS and solve the subproblem
    for s in sce:  
        # Update RHS of inventory flow balance constraints
        for i in range(V):
            for t in range(1, T):
                pi_opt_inventory_balance_U[i, t].setAttr(GRB.Attr.RHS, d_its[(i, t, s)])

        # Update RHS of initial shortage constraints
        for i in range(V):
            pi_opt_initial_shortage_U[i].setAttr(GRB.Attr.RHS, d_its[(i, 0, s)])

        # Update RHS of flow conservation constraints
        for i in range(V):
            for r in range(R):
                for t in range(T - 1):
                    pi_opt_flow_conservation[i, r, t].setAttr(GRB.Attr.RHS, I_mvp_values[i, r, t+1] - I_mvp_values[i, r, t])

        # Solve the optimality subproblem
        sub_opt.optimize()    

        if sub_opt.status == GRB.OPTIMAL:
            # Update the optimal objective value for each scenario
            mvpobj += sub_opt.objVal

    # Compute the final MVP objective value
    mvpobj /= len(sce)

    return mvpobj