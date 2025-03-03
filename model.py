import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from demand_data import DemandData

### Extended
class TwoStageSPModel:
    def __init__(self, network_processor, demand_processor, num_sce, is_mvp=False):
        self.network_processor = network_processor  # Store network data
        self.demand_processor = demand_processor  # Store demand data
        self.is_mvp = is_mvp  # Flag to differentiate between EXT and MVP models

        if self.is_mvp:
            self.num_sce = 1  # Only one scenario for MVP
            self.demand_processor.compute_expected_demand()  # Modify d_its to store expected demand
        else:
            self.num_sce = num_sce  # Keep multiple scenarios for EXT

    def ext_model(self):
        m = gp.Model("Two-Stage Stochastic Programming Model")

        V = self.network_processor.V
        V0 = self.network_processor.V0
        V1 = self.network_processor.V1
        V3 = self.network_processor.V3
        R = self.network_processor.R
        T = self.network_processor.T
        T_Bef = self.network_processor.T_Bef
        A = self.network_processor.A
        K_ijt = self.network_processor.K_ijt
        w_r = self.network_processor.w_r
        I_ir0 = self.network_processor.I_ir0
        g_shelter = self.network_processor.g_shelter
        g_pod = self.network_processor.g_pod
        t_ijt = self.network_processor.t_ijt   
        P = [1] if self.is_mvp else self.demand_processor.P  # Single scenario probability for MVP
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij
        sce = range(self.num_sce)  # Number of scenarios

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
        
        #Starting Value of I[t=0]
        for i in range(V):
            for r in range(R):
                m.addConstr(I[i,r,0]==I_ir0[i][r])
        
        # Update demand shortage for all locations, time periods (except t=0), and scenarios
        for i in range(V):
            for t in range(1, T):
                for s in sce:
                    m.addConstr(U[i, t, s] == U[i, t-1, s] - S[i, t, s] + self.demand_processor.d_its[(i, t, s)], 
                                name="DemandShortageUpdate_%s_%s_%s" % (i, t, s))
        
        # Initial demand shortage for all locations and scenarios at time period t=0
        for i in range(V):
            for s in sce:
                m.addConstr(U[i, 0, s] == self.demand_processor.d_its[(i, 0, s)], name="InitialDemandShortage_%s_%s" % (i, s))
        
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

        objective = gp.quicksum(
            P[s] * (
                gp.quicksum(
                    gp.quicksum(p_i[i] * U[i, t, s] for i in range(V)) for t in range(T)
                ) +
                gp.quicksum(
                    gp.quicksum(
                        gp.quicksum(cost_ir[i][r] * f[i, j, r, t, s] for i in V0 for j in range(V) if (i, j) in A) +
                        gp.quicksum(m_ij[i][j] * f[i, j, r, t, s] for i in range(V) for j in range(V) if (i, j) in A)
                        for r in range(R)
                    ) for t in range(T)
                )
            ) for s in sce
        )
            
        m.setObjective(objective, GRB.MINIMIZE)
        m.update()
        return m

    ### First-stage
    def master_problem(self, thetaLB):

        V = self.network_processor.V
        R = self.network_processor.R
        T = self.network_processor.T
        V0 = self.network_processor.V0
        V3 = self.network_processor.V3
        T_Bef = self.network_processor.T_Bef
        A = self.network_processor.A
        I_ir0 = self.network_processor.I_ir0
        K_ijt = self.network_processor.K_ijt
        w_r = self.network_processor.w_r

        # Create master model
        self.master_model = gp.Model("MasterProblem")
        self.master_model.modelSense = GRB.MINIMIZE

        #Define first-stage decision variables
        self.I = {}
        for i in range(V):
            for r in range(R):
                for t in range(T):
                    self.I[i, r, t] = self.master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{r}_{t}")

        #Define the auxiliary variable
        self.theta = self.master_model.addVar(vtype=GRB.CONTINUOUS, lb=thetaLB, name="theta")

        #Add constraints
        #Initial inventory levels
        for i in range(V):
            for r in range(R):
                self.master_model.addConstr(self.I[i, r, 0] == I_ir0[i][r], name=f"initial_inventory_{i}_{r}")

        #Ensure inventory at warehouses only decreases over time
        for i in V0:
            for r in range(R):
                for t in range(1, T):
                    self.master_model.addConstr(self.I[i, r, t] <= self.I[i, r, t-1], name=f"inventory_decrease_warehouse_{i}_{r}_{t}")

        #Ensure inventory at PoDs only increases before landfall
        for i in V3:
            for r in range(R):
                for t in T_Bef[1:]:
                    self.master_model.addConstr(self.I[i, r, t] >= self.I[i, r, t-1], name=f"inventory_increase_pod_{i}_{r}_{t}")

        #Lower bound of inventory at warehouses (V0)
        for i in V0:
            for r in range(R):
                for t in range(1, T): 
                    # Calculate the total arc capacity for the current time period
                    arc_capacity = sum(K_ijt[i][j][t] for j in range(V) if (i, j) in A)
                    # Calculate max outflow for commodity r
                    max_outflow_r = arc_capacity  / w_r[r]
                    if I_ir0[i][r] >= max_outflow_r:
                        self.master_model.addConstr(self.I[i, r, t] >= I_ir0[i][r] - max_outflow_r, name=f"LowerBound_Warehouse_{i}_{r}_{t}")

        #Upper bound of inventory at PoDs (V3)
        for i in V3:
            for r in range(R):
                for t in T_Bef[1:]:
                    # Calculate the total arc capacity for the current time period
                    arc_capacity = sum(K_ijt[j][i][t] for j in range(V) if (j, i) in A)
                    # Calculate max inflow for commodity r
                    max_inflow_r = arc_capacity / w_r[r]            
                    self.master_model.addConstr(self.I[i, r, t] <= I_ir0[i][r] + max_inflow_r, name=f"UpperBound_PoD_{i}_{r}_{t}")

        self.master_model.setObjective(self.theta, GRB.MINIMIZE)
        self.master_model.setParam("OutputFlag", 0)
        self.master_model.update()

    def optimality_sub(self):
        V = self.network_processor.V
        R = self.network_processor.R
        T = self.network_processor.T
        A = self.network_processor.A
        V0 = self.network_processor.V0
        V1 = self.network_processor.V1
        V3 = self.network_processor.V3
        t_ijt = self.network_processor.t_ijt
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij
        g_shelter = self.network_processor.g_shelter
        g_pod = self.network_processor.g_pod
        w_r = self.network_processor.w_r
        K_ijt = self.network_processor.K_ijt

        # Create optimality subproblem model
        self.sub_opt = gp.Model("OptimalitySubproblem")
        self.sub_opt.modelSense = GRB.MINIMIZE

        #Define decision variables
        self.U_opt = {}
        self.S_opt = {}
        self.f_opt = {}
        
        # U: Demand shortage at location i at time period t in scenario s
        for i in range(V):
            for t in range(T):
                self.U_opt[i, t] = self.sub_opt.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_opt_{i}_{t}")
        
        # S: Demand satisfied at location i at time period t in scenario s
        for i in range(V):
            for t in range(T):
                self.S_opt[i, t] = self.sub_opt.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_opt_{i}_{t}")
        
        # f: Quantity of commodity r traveling on link (i, j) at time period t in scenario s
        for i in range(V):
            for j in range(V):
                if (i, j) in A:
                    for r in range(R):
                        for t in range(T):
                            self.f_opt[i, j, r, t] = self.sub_opt.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_opt_{i}_{j}_{r}_{t}")

        #Objective function
        self.sub_opt.setObjective(
            gp.quicksum(  # Demand shortage cost
                gp.quicksum(p_i[i] * self.U_opt[i, t] for i in range(V))
                for t in range(T)) +
            gp.quicksum(  # Procurement and transportation costs
                gp.quicksum(
                    gp.quicksum(
                        cost_ir[i][r] * self.f_opt[i, j, r, t]  # Procurement cost at V0
                        for i in V0 for j in range(V) if (i, j) in A) +
                    gp.quicksum(
                        m_ij[i][j] * self.f_opt[i, j, r, t]  # Transportation cost for all locations
                        for i in range(V) for j in range(V) if (i, j) in A) 
                    for r in range(R)) for t in range(T)))

        #Constraints
        # Constraint for inventory flow balance with U and S
        self.pi_opt_inventory_balance_U = {}
        for i in range(V):
            for t in range(1, T):  # Start from t=1 since t=0 is the initial condition
                self.pi_opt_inventory_balance_U[i, t] = self.sub_opt.addConstr(self.U_opt[i, t] - self.U_opt[i, t-1] + self.S_opt[i, t] == 0, 
                                                                    name=f"inventory_balance_U_opt_{i}_{t}")
        
        # Initial shortage for demand at t=0
        self.pi_opt_initial_shortage_U = {}
        for i in range(V):
            self.pi_opt_initial_shortage_U[i] = self.sub_opt.addConstr(self.U_opt[i, 0] == 0, name=f"initial_shortage_U_opt_{i}")
        
        # Flow conservation constraint for f with the RHS set to 0 temporarily, using appropriate g_shelter or g_pod
        self.pi_opt_flow_conservation = {}
        for i in range(V):
            for r in range(R):
                for t in range(T - 1):
                    # Add the constraint to the model
                    self.pi_opt_flow_conservation[i, r, t] = self.sub_opt.addConstr(
                        -gp.quicksum(self.f_opt[i, j, r, t] for j in range(V) if (i, j) in A)
                        + gp.quicksum(self.f_opt[j, i, r, t - t_ijt[j][i] + 1] for j in range(V) if (i, j) in A and t - t_ijt[j][i] + 1 >= 0)
                        - self.S_opt[i, t] * (g_shelter[r] if i in V1 else g_pod[r] if i in V3 else 0) == 0,
                        name=f"flow_conservation_opt_{i}_{r}_{t}")
        
        # Arc capacity constraint
        self.pi_opt_arc_capacity = {}
        for i in range(V):
            for j in range(V):
                if (i, j) in A:  # Only consider valid links in A
                    for t in range(T):
                            self.pi_opt_arc_capacity[i, j, t] = self.sub_opt.addConstr(-gp.quicksum(w_r[r] * self.f_opt[i, j, r, t] for r in range(R)) >= -K_ijt[i][j][t],
                                name=f"arc_capacity_opt_{i}_{j}_{t}")

        self.sub_opt.setParam("OutputFlag", 0)

    def feasibility_sub(self):
        V = self.network_processor.V
        R = self.network_processor.R
        T = self.network_processor.T
        A = self.network_processor.A
        V1 = self.network_processor.V1
        V3 = self.network_processor.V3
        t_ijt = self.network_processor.t_ijt
        g_shelter = self.network_processor.g_shelter
        g_pod = self.network_processor.g_pod
        w_r = self.network_processor.w_r
        K_ijt = self.network_processor.K_ijt

        # Create feasibility subproblem model
        self.sub_feas = gp.Model("FeasibilitySubproblem")
        self.sub_feas.modelSense = GRB.MINIMIZE

        #Define decision variables
        self.U_feas = {}
        self.S_feas = {}
        self.f_feas = {}
        
        # U: Demand shortage at location i at time period t in scenario s
        for i in range(V):
            for t in range(T):
                self.U_feas[i, t] = self.sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f"U_feas_{i}_{t}")
        
        # S: Demand satisfied at location i at time period t in scenario s
        for i in range(V):
            for t in range(T):
                self.S_feas[i, t] = self.sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f"S_feas_{i}_{t}")
        
        # f: Quantity of commodity r traveling on link (i, j) at time period t in scenario s
        for i in range(V):
            for j in range(V):
                if (i, j) in A:
                    for r in range(R):
                        for t in range(T):
                            self.f_feas[i, j, r, t] = self.sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f"f_feas_{i}_{j}_{r}_{t}")

        #Define slack variables ε (epsilon) for each group in the objective function
        self.epsilon_1 = {}
        for i in range(V):
            for r in range(R):
                for t in range(T-1):
                    self.epsilon_1[i, r, t] = self.sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"epsilon1_{i}_{r}_{t}")
        
        self.epsilon_2 = {}
        for i in range(V):
            for r in range(R):
                for t in range(T-1):
                    self.epsilon_2[i, r, t] = self.sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"epsilon2_{i}_{r}_{t}")
        
        self.epsilon_3 = {}
        for i in range(V):
            for j in range(V):
                if (i, j) in A:
                    for t in range(T):
                        self.epsilon_3[i, j, t] = self.sub_feas.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"epsilon3_{i}_{j}_{t}")

        self.sub_feas.setObjective(
        gp.quicksum(self.epsilon_1[i, r, t] + self.epsilon_2[i, r, t] for i in range(V) for r in range(R) for t in range(T - 1)) +
        gp.quicksum(self.epsilon_3[i, j, t] for i in range(V) for j in range(V) if (i, j) in A for t in range(T)))

        # Constraints
        # Constraint: Inventory flow balance for U, S, and epsilon variables
        self.pi_feas_inventory_balance_U = {}
        for i in range(V):
            for t in range(1, T):  # Note: starting from t=1 as per the constraint
                self.pi_feas_inventory_balance_U[i, t] = self.sub_feas.addConstr(
                    self.U_feas[i, t] - self.U_feas[i, t-1] + self.S_feas[i, t] == 0,
                    name=f"inventory_balance_U_feas_{i}_{t}")
        
        # Initialize shortage for U at t=0
        self.pi_feas_initial_shortage_U = {}
        for i in range(V):
            self.pi_feas_initial_shortage_U[i] = self.sub_feas.addConstr(self.U_feas[i, 0] == 0, name=f"initial_shortage_U_feas_{i}")
        
        # Flow conservation constraint
        self.pi_feas_flow_conservation = {}
        for i in range(V):
            for r in range(R):
                for t in range(T - 1):
                    self.pi_feas_flow_conservation[i, r, t] = self.sub_feas.addConstr(
                        -gp.quicksum(self.f_feas[i, j, r, t] for j in range(V) if (i, j) in A)
                        + gp.quicksum(self.f_feas[j, i, r, t - t_ijt[j][i] + 1] for j in range(V) if (i, j) in A and t - t_ijt[j][i] + 1 >= 0)
                        - self.S_feas[i, t] * (g_shelter[r] if i in V1 else g_pod[r] if i in V3 else 0)
                        + self.epsilon_1[i, r, t] - self.epsilon_2[i, r, t] == 0,
                        name=f"flow_conservation_feas_{i}_{r}_{t}")
        
        # Flow capacity constraint for all links, time periods, and scenarios
        self.pi_feas_arc_capacity = {}
        for i in range(V):
            for j in range(V):
                if (i, j) in A:  # Only consider valid links in A
                    for t in range(T):
                        self.pi_feas_arc_capacity[i, j, t] = self.sub_feas.addConstr(
                            -gp.quicksum(w_r[r] * self.f_feas[i, j, r, t] for r in range(R))
                            + self.epsilon_3[i, j, t] >= -K_ijt[i][j][t],
                            name=f"arc_capacity_feas_{i}_{j}_{t}")

        # Disable Gurobi solver output
        self.sub_feas.setParam("OutputFlag", 0)

    def compute_mvp_objective(self, I_mvp_values):
        # Recreate demand data processor
        demand_processor = DemandData(self.demand_processor.num_sce, self.demand_processor.instance_version)
        demand_processor.generate_scenario_data()
        demand_processor.load_demand_data()  # Regenerate d_its from scratch
        P = demand_processor.P  # Get new scenario probabilities
        d_its = demand_processor.d_its  # Use reloaded d_its instead of the modified one
        P = self.demand_processor.P
        V = self.network_processor.V
        R = self.network_processor.R
        T = self.network_processor.T
        sce = range(len(P))
        mvpobj = 0  # Initialize the MVP objective value

        # Loop over all scenarios to update RHS and solve the subproblem
        for s in sce:  
            # Update RHS of inventory flow balance constraints
            for i in range(V):
                for t in range(1, T):
                    self.pi_opt_inventory_balance_U[i, t].setAttr(GRB.Attr.RHS, d_its[(i, t, s)])

            # Update RHS of initial shortage constraints
            for i in range(V):
                self.pi_opt_initial_shortage_U[i].setAttr(GRB.Attr.RHS, d_its[(i, 0, s)])

            # Update RHS of flow conservation constraints
            for i in range(V):
                for r in range(R):
                    for t in range(T - 1):
                        self.pi_opt_flow_conservation[i, r, t].setAttr(GRB.Attr.RHS, I_mvp_values[i, r, t+1] - I_mvp_values[i, r, t])

            # Solve the optimality subproblem
            self.sub_opt.optimize()    

            if self.sub_opt.status == GRB.OPTIMAL:
                # Update the optimal objective value for each scenario
                mvpobj += self.sub_opt.objVal

        # Compute the final MVP objective value
        mvpobj /= len(sce)

        return mvpobj
