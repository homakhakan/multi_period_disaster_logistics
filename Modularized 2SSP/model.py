import gurobipy as gp
from gurobipy import GRB

def create_ext(sets, parameters, d_its, num_sce):
    """
    Creates the Extended Formulation Model.
    """
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
    """
    Defines decision variables for the Extended Formulation model.
    """
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
    """
    Defines constraints for the Extended Formulation model.
    """
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
    """
    Defines the objective function for the Extended Formulation model.
    """
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