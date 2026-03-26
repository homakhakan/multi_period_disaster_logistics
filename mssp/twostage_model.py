import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from demand_data import DemandData
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from demand_gen_indep import new_in_sample

class TwoStageSPModel:
    def __init__(self, network_processor, demand_processor, num_sce, num_sce_mean=None):
        self.network_processor = network_processor  # Store network data
        self.demand_processor = demand_processor  # Store demand data
        self.num_sce = num_sce  # Use full scenarios for other models
        self.num_sce_mean = num_sce_mean # Deterministic mean value RH
        self.fixed_flow = {} # Initialize fixed_flow for use across RH-related functions

    def ext_model(self, is_mvp=False, model_version=1):

        if is_mvp:
            self.demand_processor.d_its = self.demand_processor.d_its_expected
            self.demand_processor.P = [1]
            self.num_sce = 1
        else:
            self.num_sce = len(self.demand_processor.P)

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
        C_i0 = self.network_processor.C_i0
        h_i = self.network_processor.h_i
        g_shelter = self.network_processor.g_shelter
        g_pod = self.network_processor.g_pod
        t_ij = self.network_processor.t_ij   
        P = self.demand_processor.P
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij
        sce = range(self.num_sce)  # Number of scenarios

        # Define first-stage decision variables
        # I: Inventory of commodity r at node i at the start of time period t
        I = {}
        
        for i in range(V):
            for r in range(R):
                for t in range(T):
                    I[i, r, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="I_%s_%s_%s" % (i, r, t))

        # Add vehicle routing variables if model version 2 is selected
        if model_version == 2:
            K = {}  # Number of vehicles on arc (i,j) at time t
            for i in range(V):
                for j in range(V):
                    if (i, j) in A:
                        for t in range(T):
                            K[i, j, t] = m.addVar(vtype=GRB.INTEGER, lb=0, name="K_%s_%s_%s" % (i, j, t))

            J = {}  # Number of vehicles at node i at time t
            for i in range(V):
                for t in range(T):
                    J[i, t] = m.addVar(vtype=GRB.INTEGER, lb=0, name="J_%s_%s" % (i, t))
        
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

        if model_version == 1:
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
                for s in sce:
                    m.addConstr(I[i, r, 0] == I_ir0[i][r] - gp.quicksum(f[i, j, r, 0, s] for j in range(V) if (i, j) in A) - S[i, 0, s]) ## WATCH OUT for g_r
        
        # Update demand shortage for all locations, time periods (except t=0), and scenarios
        for i in range(V):
            for t in range(1, T):
                for s in sce:
                    m.addConstr(U[i, t, s] == U[i, t-1, s] - S[i, t, s] + self.demand_processor.d_its[(i, t, s)], 
                                name="DemandShortageUpdate_%s_%s_%s" % (i, t, s))
        
        # Initial demand shortage for all locations and scenarios at time period t=0
        for i in range(V):
            for s in sce:
                m.addConstr(U[i, 0, s] == self.demand_processor.d_its[(i, 0, s)] - S[i, 0, s], name="InitialDemandShortage_%s_%s" % (i, s))
        
        # Add nonanticipativity constraints
        for (i, j) in A:
            for r in range(R):
                for s in range(1, self.num_sce): # skip s=0 (used as reference) 
                    m.addConstr(f[i, j, r, 0, s] == f[i, j, r, 0, 0], name=f"NonAntic_f_{i}_{j}_{r}_{0}_{s}")
        
        # Inventory update constraint for all locations, commodities, time periods (except 0), and scenarios
        for i in range(V):
            for r in range(R):
                for t in range(1, T):
                    for s in sce:
                        outgoing_flow = gp.quicksum(f[i, j, r, t, s] for j in range(V) if (i, j) in A)
                        incoming_flow = gp.quicksum(f[j, i, r, t - (t_ij[j][i] + 1), s] for j in range(V) if (j, i) in A and t - (t_ij[j][i] + 1) >= 0)
                        
                        # Inventory update constraint for shelters
                        if i in V1:
                            m.addConstr(
                                I[i, r, t] == I[i, r, t-1] - outgoing_flow + incoming_flow - S[i, t, s] * g_shelter[r],
                                name="InventoryUpdate_Shelter_%s_%s_%s_%s" % (i, r, t, s)
                            )
                        # Inventory update constraint for pods
                        elif i in V3:
                            m.addConstr(
                                I[i, r, t] == I[i, r, t-1] - outgoing_flow + incoming_flow - S[i, t, s] * g_pod[r],
                                name="InventoryUpdate_Pod_%s_%s_%s_%s" % (i, r, t, s)
                            )
                        # Inventory update constraint for warehouses and RSAs
                        else:
                            m.addConstr(
                                I[i, r, t] == I[i, r, t-1] - outgoing_flow + incoming_flow,
                                name="InventoryUpdate_%s_%s_%s_%s" % (i, r, t, s)
                            )
        
        # Flow capacity constraint for all links, time periods, and scenarios
        for i in range(V):
            for j in range(V):
                if (i, j) in A:  # Only consider valid links in A
                    for t in range(T):
                        for s in sce:
                                
                            if model_version == 2:
                                m.addConstr(
                                gp.quicksum(w_r[r] * f[i, j, r, t, s] for r in range(R)) <= K[i, j, t],
                                name="FlowCapacity_%s_%s_%s_%s" % (i, j, t, s))
                            else:
                                m.addConstr(
                                gp.quicksum(w_r[r] * f[i, j, r, t, s] for r in range(R)) <= K_ijt[i][j][t],
                                name="FlowCapacity_%s_%s_%s_%s" % (i, j, t, s))

        if model_version == 2:
            # Vehicle initializtion constraint
            for i in range(V):
                m.addConstr(J[i, 0] == C_i0[i], name="VehicleInit_%s" % i)

            
            # Vehicle flow balance constraint
            for i in range(V):
                for t in range(T - 1):
                    outgoing = gp.quicksum(K[i, j, t] for j in range(V) if (i, j) in A)
                    incoming = gp.LinExpr()
                    for j in range(V):
                        if (j, i) in A:
                            t_prime = t - t_ij[j][i] - 1 - h_i[i]
                            if t_prime >= 0:
                                incoming += K[j, i, t_prime]
                    m.addConstr(
                        J[i, t + 1] == J[i, t] - outgoing + incoming, name="VehicleCont_%s_%s" % (i, t))

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
        if model_version == 2:
            m.setParam("Heuristics", 0.5)       # Increase heuristic effort
            m.setParam("MIPFocus", 1)           # Focus on finding feasible solutions
            m.setParam("Cuts", 2)               # Try more aggressive cutting planes
            m.setParam("Presolve", 2)           # Aggressive presolve
            m.setParam("MIPGap", 0.1)           # Stop when MIP gap reaches 10%

        m.setParam("OutputFlag", 0)
        m.update()
        return m, I

    def OOS_2SSP(self, fixed_I_irt, args):
        """
        Evaluate the deterministic second-stage cost for a single OOS demand scenario,
        by fixing all inventory variables I[i, r, t] to values from the static solution.

        Args:
            fixed_I_irt (dict): Dictionary of inventory values {(i, r, t): value}

        Returns:
            float: Total cost from solving the model
        """
        # Setup for deterministic single-scenario evaluation
        self.num_sce = 1
        self.demand_processor.P = [1]

        # Build appropriate model depending on the method
        if args.solution_method == 1:
            model, I_vars = self.ext_model(is_mvp=False)
        elif args.solution_method == 8:
            model, I_vars = self.WS_model(args.t_tilda)
        else:
            raise ValueError("OOS evaluation only supported for methods 1 (2SSP) and 8 (WS).")

        # Fix each I[i, r, t] to its corresponding value using equality constraints
        for (i, r, t), val in fixed_I_irt.items():
            if abs(val) < 1e-5:
                val = 0
            model.addConstr(I_vars[i, r, t] == val, name=f"fix_I_{i}_{r}_{t}")

        # Solve the model
        model.optimize()

        # Return objective value if solved successfully
        if model.status == GRB.OPTIMAL:
            return model.objVal
        else:
            model.write("non_optimal_model.lp")
            print(f"[Warning] Model not optimal. Status: {model.status}")
            return float("inf")

    def extract_all_inventory(self, I_vars, t_tilda=0):
        inventory_values = {}
        for i in range(self.network_processor.V):
            for r in range(self.network_processor.R):
                for t in range(t_tilda, self.network_processor.T):
                    inventory_values[(i, r, t)] = I_vars[i, r, t].X
        return inventory_values
    
    def WS_model(self, t_tilda):

        m = gp.Model(f"Wait-and-See_model_t_tilda={t_tilda} ") # Based on 2SSP

        self.num_sce = len(self.demand_processor.P)
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
        C_i0 = self.network_processor.C_i0
        h_i = self.network_processor.h_i
        g_shelter = self.network_processor.g_shelter
        g_pod = self.network_processor.g_pod
        t_ij = self.network_processor.t_ij 
        P = self.demand_processor.P
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij
        sce = range(self.num_sce)  # Number of scenarios

        # Define first-stage decision variables
        # I: Inventory of commodity r at node i starting from t_tilda
        I = {}
        
        for i in range(V):
            for r in range(R):
                for t in range(t_tilda, T):
                    I[i, r, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="I_%s_%s_%s" % (i, r, t))

        # Define second-stage decision variables
        U = {}  # Demand shortage at location i starting from t_tilda in scenario s
        
        for i in range(V):
            for t in range(t_tilda, T):
                for s in sce:  # Iterate directly over the scenario set (e.g., 0 and 1)
                    U[i, t, s] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="U_%s_%s_%s" % (i, t, s))

        S = {}  # Demand satisfied at location i starting from t_tilda in scenario s
        
        for i in range(V):
            for t in range(t_tilda, T):
                for s in sce:  # Iterate directly over the scenario set
                    S[i, t, s] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S_%s_%s_%s" % (i, t, s))
        
        f = {}  # Quantity of commodity r traveling on link (i, j) starting from t_tilda in scenario s
        
        for i in range(V):
            for j in range(V):
                if (i, j) in A:  # Only define f_ijrt^s if (i, j) is a valid link
                    for r in range(R):
                        for t in range(t_tilda, T):
                            for s in sce:  # Iterate directly over the scenario set
                                f[i, j, r, t, s] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_%s_%s_%s_%s_%s" % (i, j, r, t, s))
        m.update()

        # Starting Value of I[t = t_tilda]
        for i in range(V):
            for r in range(R):
                for s in sce:
                    m.addConstr(I[i, r, t_tilda] == I_ir0[i][r] - gp.quicksum(f[i, j, r, t_tilda, s] for j in range(V) if (i, j) in A) - S[i, t_tilda, s]) ## WATCH OUT for g_r

        # Initial demand shortage for all locations and scenarios at time period t=t_tilda
        for i in range(V):
            for s in sce:
                past_demand = sum(self.demand_processor.d_its[(i, t, s)] for t in range(t_tilda))
                m.addConstr(U[i, t_tilda, s] == self.demand_processor.d_its[(i, t_tilda, s)] - S[i, t_tilda, s], name="InitialDemandShortage_%s_%s" % (i, s))

        # Update demand shortage for all locations, time periods (except t = t_tilda), and scenarios
        for i in range(V):
            for t in range(t_tilda + 1, T):
                for s in sce:
                    m.addConstr(U[i, t, s] == U[i, t-1, s] - S[i, t, s] + self.demand_processor.d_its[(i, t, s)], 
                                name="DemandShortageUpdate_%s_%s_%s" % (i, t, s))
                    
        # Add nonanticipativity constraints
        for (i, j) in A:
            for r in range(R):
                for s in range(1, self.num_sce): # skip s=0 (used as reference) 
                    m.addConstr(f[i, j, r, t_tilda, s] == f[i, j, r, t_tilda, 0], name=f"NonAntic_f_{i}_{j}_{r}_{t_tilda}_{s}")

        # Inventory update constraint for all locations, commodities, time periods (except t_tilda), and scenarios
        for i in range(V):
            for r in range(R):
                for t in range(t_tilda + 1, T):
                    for s in sce:
                        outgoing_flow = gp.quicksum(f[i, j, r, t, s] for j in range(V) if (i, j) in A)
                        incoming_flow = gp.quicksum(f[j, i, r, t - (t_ij[j][i] + 1), s] for j in range(V) if (j, i) in A and t - (t_ij[j][i] + 1) >= t_tilda)
                        
                        # Inventory update constraint for shelters
                        if i in V1:
                            m.addConstr(
                                I[i, r, t] == I[i, r, t-1] - outgoing_flow + incoming_flow - S[i, t, s] * g_shelter[r],
                                name="InventoryUpdate_Shelter_%s_%s_%s_%s" % (i, r, t, s)
                            )
                        # Inventory update constraint for pods
                        elif i in V3:
                            m.addConstr(
                                I[i, r, t] == I[i, r, t-1] - outgoing_flow + incoming_flow - S[i, t, s] * g_pod[r],
                                name="InventoryUpdate_Pod_%s_%s_%s_%s" % (i, r, t, s)
                            )
                        # Inventory update constraint for warehouses and RSAs
                        else:
                            m.addConstr(
                                I[i, r, t] == I[i, r, t-1] - outgoing_flow + incoming_flow,
                                name="InventoryUpdate_%s_%s_%s_%s" % (i, r, t, s)
                            )

        # Flow capacity constraint for all links, starting from t_tilda, and scenarios
        for i in range(V):
            for j in range(V):
                if (i, j) in A:  # Only consider valid links in A
                    for t in range(t_tilda, T):
                        for s in sce:
                            m.addConstr(gp.quicksum(w_r[r] * f[i, j, r, t, s] for r in range(R)) <= K_ijt[i][j][t], 
                                        name="FlowCapacity_%s_%s_%s_%s" % (i, j, t, s))

        m.update()

        objective = gp.quicksum(
            P[s] * (
                gp.quicksum(
                    gp.quicksum(p_i[i] * U[i, t, s] for i in range(V)) for t in range(t_tilda, T)
                ) +
                gp.quicksum(
                    gp.quicksum(
                        gp.quicksum(cost_ir[i][r] * f[i, j, r, t, s] for i in V0 for j in range(V) if (i, j) in A) +
                        gp.quicksum(m_ij[i][j] * f[i, j, r, t, s] for i in range(V) for j in range(V) if (i, j) in A)
                        for r in range(R)
                    ) for t in range(t_tilda, T)
                )
            ) for s in sce
        )

        m.setObjective(objective, GRB.MINIMIZE)

        m.setParam("OutputFlag", 0)
        m.update()

        return m, I

    def RH_model_creator(self, t_bar, sol):

        # Builds the Gurobi model for the current rolling horizon window starting at time t_bar.
        # Returns the model and the key variable dictionaries (I, U, S, f).

        V = self.network_processor.V
        V0 = self.network_processor.V0
        V1 = self.network_processor.V1
        V3 = self.network_processor.V3
        R = self.network_processor.R
        T = self.network_processor.T
        A = self.network_processor.A
        K_ijt = self.network_processor.K_ijt
        w_r = self.network_processor.w_r
        I_ir0 = self.network_processor.I_ir0
        g_shelter = self.network_processor.g_shelter
        g_pod = self.network_processor.g_pod
        t_ij = self.network_processor.t_ij  
        P = self.demand_processor.P
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij
        sce = range(self.num_sce)  # Number of scenarios

        model = gp.Model("Rolling Horizon Model")

        # I: Inventory of commodity r at node i at the start of time period t
        I = {}
        
        for i in range(V):
            for r in range(R):
                for t in range(t_bar, T):
                    I[i, r, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{r}_{t}")
        
        # Define second-stage decision variables
        U = {}  # Demand shortage at location i at time period t in scenario s
        
        for i in range(V):
            for t in range(t_bar, T):
                for s in sce:  # Iterate directly over the scenario set (e.g., 0 and 1)
                    U[i, t, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_{i}_{t}_{s}")
        
        S = {}  # Demand satisfied at location i at time period t in scenario s
        
        for i in range(V):
            for t in range(t_bar, T):
                for s in sce:  # Iterate directly over the scenario set
                    S[i, t, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{i}_{t}_{s}")
        
        f = {}  # Quantity of commodity r traveling on link (i, j) at time period t in scenario s
        
        for (i, j) in A:  # Only define if (i, j) is a valid link
            for r in range(R):
                for t in range(t_bar, T):
                    for s in sce:  # Iterate directly over the scenario set
                        f[i, j, r, t, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}_{r}_{t}_{s}")

        if t_bar == 0:
            #Starting Value of I[t=0]
            for i in range(V):
                for r in range(R):
                    for s in sce:
                        model.addConstr(I[i, r, 0] == I_ir0[i][r] - gp.quicksum(f[i, j, r, 0, s] for j in range(V) if (i, j) in A) - S[i, 0, s],
                                        name=f"InitialInventory_{i}_{r}_{s}") ## WATCH OUT for g_r

            # Initial demand shortage for all locations and scenarios at time period t=0
            for i in range(V):
                for s in sce:
                    model.addConstr(U[i, 0, s] == self.demand_processor.d_its[(i, 0)] - S[i, 0, s], name=f"InitialDemandShortage_{i}_{s}")

        else:
            # # Updated initial inventory
            # for i in range(V):
            #     for r in range(R):
            #         model.addConstr(I[i, r, t_bar] == sol["I"][i, r, t_bar - 1], name=f"InitInv_{i}_{r}_{t_bar-1}")
                    
            # Initialize unmet demand at t_bar using realized demand and satisfied demand S.       
            for i in range(V):
                for s in sce:
                    model.addConstr(U[i, t_bar, s] == sol["U"][i, t_bar - 1] + self.demand_processor.d_its[(i, t_bar)] - S[i, t_bar, s], name=f"InitUnmet_{i}_{s}_{t_bar}")

            # Updated flow for all locations, commodities, and scenarios only for t_bar
            for i in range(V):
                for r in range(R):
                    for s in sce:
                        outgoing_flow = gp.quicksum(f[i, j, r, t_bar, s] for j in range(V) if (i, j) in A)
                        
                        # Regular incoming flow (from inside RH window)
                        incoming_flow_regular = gp.quicksum(f[j, i, r, t_prime, s] for j in range(V) for t_prime in range(t_bar, T)
                                                            if (j, i) in A and t_prime == t_bar - (t_ij[j][i] + 1))
                        
                        incoming_flow_fixed = gp.quicksum(self.fixed_flow[(j, i, r, t_prime)] for j in range(V) for t_prime in range(t_bar)
                                                            if (j, i) in A and t_prime == t_bar - (t_ij[j][i] + 1))

                        total_incoming = incoming_flow_regular + incoming_flow_fixed

                        # Inventory update constraint for shelters
                        if i in V1:
                            model.addConstr(I[i, r, t_bar] == sol["I"][i, r, t_bar - 1] - outgoing_flow + total_incoming - S[i, t_bar, s] * g_shelter[r], 
                                            name=f"InventoryUpdate_Shelter_{i}_{r}_{t_bar}_{s}")
                        # Inventory update constraint for pods
                        elif i in V3:
                            model.addConstr(I[i, r, t_bar] == sol["I"][i, r, t_bar - 1] - outgoing_flow + total_incoming - S[i, t_bar, s] * g_pod[r],
                                            name=f"InventoryUpdate_Pod_{i}_{r}_{t_bar}_{s}")
                        # Inventory update constraint for warehouses and RSAs
                        else:
                            model.addConstr(I[i, r, t_bar] == sol["I"][i, r, t_bar - 1] - outgoing_flow + total_incoming, 
                                            name=f"InventoryUpdate_{i}_{r}_{t_bar}_{s}")

                    # # Debug print only when t_bar == 6
                    # if t_bar == 5:
                    #     d_val = self.demand_processor.d_its[(i, t_bar)]
                    #     s_val = sol["S"][i, t_bar]
                    #     print(f"t_bar={t_bar}, i={i}, s={s}: d_its={d_val}, S={s_val}, U = {d_val - s_val}")
                                
        # Updated flow for all locations, commodities, time periods (except last), and scenarios
        for i in range(V):
            for r in range(R):
                for t in range(t_bar + 1, T):
                    for s in sce:
                        outgoing_flow = gp.quicksum(f[i, j, r, t, s] for j in range(V) if (i, j) in A)
                        
                        # Regular incoming flow (from inside RH window)
                        incoming_flow_regular = gp.quicksum(f[j, i, r, t_prime, s] for j in range(V) for t_prime in range(t_bar, T)
                                                            if (j, i) in A and t_prime == t - (t_ij[j][i] + 1))
                        
                        incoming_flow_fixed = gp.quicksum(self.fixed_flow[(j, i, r, t_prime)] for j in range(V) for t_prime in range(t_bar)
                                                            if (j, i) in A and t_prime == t - (t_ij[j][i] + 1))

                        total_incoming = incoming_flow_regular + incoming_flow_fixed

                        # Inventory update constraint for shelters
                        if i in V1:
                            model.addConstr(I[i, r, t] == I[i, r, t - 1] - outgoing_flow + total_incoming - S[i, t, s] * g_shelter[r], 
                                            name=f"InventoryUpdate_Shelter_{i}_{r}_{t}_{s}")
                        # Inventory update constraint for pods
                        elif i in V3:
                            model.addConstr(I[i, r, t] == I[i, r, t - 1] - outgoing_flow + total_incoming - S[i, t, s] * g_pod[r],
                                            name=f"InventoryUpdate_Pod_{i}_{r}_{t}_{s}")
                        # Inventory update constraint for warehouses and RSAs
                        else:
                            model.addConstr(I[i, r, t] == I[i, r, t - 1] - outgoing_flow + total_incoming, 
                                            name=f"InventoryUpdate_{i}_{r}_{t}_{s}")    

        # Update demand shortage for all locations, time periods (except t=0), and scenarios
        for i in range(V):
            for t in range(t_bar + 1, T):
                for s in sce:
                    model.addConstr(U[i, t, s] == U[i, t - 1, s] - S[i, t, s] + self.demand_processor.d_its[(i, t, s)], name=f"DemandShortageUpdate_{i}_{t}_{s}")

                    
        # Flow capacity constraint for all links, time periods, and scenarios
        for (i, j) in A:  # Only consider valid links in A
            for t in range(t_bar, T):
                for s in sce:
                    model.addConstr(gp.quicksum(w_r[r] * f[i, j, r, t, s] for r in range(R)) <= K_ijt[i][j][t], name=f"FlowCapacity_{i}_{j}_{t}_{s}")
                                
        # Add nonanticipativity constraints: all scenarios must match scenario 0 at t_bar
        for (i, j) in A:
            for r in range(R):
                for s in range(1, self.num_sce): # skip s=0 (used as reference) 
                    model.addConstr(f[i, j, r, t_bar, s] == f[i, j, r, t_bar, 0], name=f"NonAntic_f_{i}_{j}_{r}_{t_bar}_{s}")
                                
        objective = gp.quicksum(P[s] * 
                                (gp.quicksum(gp.quicksum(p_i[i] * U[i, t, s] for i in range(V)) for t in range(t_bar, T)) + 
                                 gp.quicksum(gp.quicksum(gp.quicksum(cost_ir[i][r] * f[i, j, r, t, s] for i in V0 for j in range(V) if (i, j) in A) + 
                                                         gp.quicksum(m_ij[i][j] * f[i, j, r, t, s] for i in range(V) for j in range(V) if (i, j) in A) for r in range(R))
                                                         for t in range(t_bar, T))) for s in sce)
            
        model.setObjective(objective, GRB.MINIMIZE)
        model.setParam("OutputFlag", 0)

        return model, I, U, S, f

    def RH_model_solver(self, model, variables):
        """
        Solves the given Gurobi rolling horizon model and extracts solution values.
        Returns the objective value and a dictionary containing the optimized values of decision variables (I, U, S, f). 
        If the model is not solved to optimality, returns None for both outputs.
        """
        # Solve the model
        model.optimize()

        # Check solution status
        if model.status != GRB.OPTIMAL:
            print(f"Warning: Optimization ended with status {model.status}")
            if model.status == GRB.INFEASIBLE:
                model.computeIIS()
                model.write("conflict.ilp")     # Only conflicting constraints
                model.write("debug_model.lp")   # Full LP model
                print("Wrote conflict.ilp and debug_model.lp")
            exit(0)

        # Extract objective value
        obj_val = model.objVal

        # Extract values of variables into a dictionary
        # Determine if this is terminal model (U keys are 2-tuples)
        is_terminal = all(len(k) == 2 for k in variables["U"].keys())

        if is_terminal:
            variable_values = {
                "I": {(i, r, t): var.X for (i, r, t), var in variables["I"].items()}, # variables["I"] is a dictionary of decision variables (not the optimal values!)
                "U": {(i, t): var.X for (i, t), var in variables["U"].items()},
                "S": {(i, t): var.X for (i, t), var in variables["S"].items()},
                "f": {(i, j, r, t): var.X for (i, j, r, t), var in variables["f"].items()},
            }
        else:
            variable_values = {
                "I": {(i, r, t): var.X for (i, r, t), var in variables["I"].items()},
                "U": {(i, t, s): var.X for (i, t, s), var in variables["U"].items()},
                "S": {(i, t, s): var.X for (i, t, s), var in variables["S"].items()},
                "f": {(i, j, r, t, s): var.X for (i, j, r, t, s), var in variables["f"].items()},
            }

        # Return everything
        return obj_val, variable_values

    def RH_sol_creator(self, variable_values, t_bar):
        """
        Extracts inventory and unmet demand values at time t_bar + 1 from the solved model for the realized scenario. 
        Updates the solution dictionary (sol) with these values and returns it
        Used for initializing the next rolling horizon stage.
        """
        sol = {}

        # Extract inventory at time t_bar (end of current roll) to use as initial inventory in next roll
        sol["I"] = {(i, r, t_bar): val for (i, r, t), val in variable_values["I"].items() if t == t_bar}

        # Extract satisfied demand S at time t = t_bar + 1 (start of next roll).
        # Since S_it^s is scenario-independent only at the first time period of each roll (i.e., S_it^s = S_it ∀s),
        # we only store one copy from scenario s = 0.
        # sol["S"] = {(i, next_t): val for (i, t, s), val in variable_values["S"].items() if t == t_bar and s == 0}

        sol["U"] = {(i, t_bar): val for (i, t, s), val in variable_values["U"].items() if t == t_bar and s == 0}

        # # Print for debugging
        # if next_t == 4:
        # for (i, t, s), val in variable_values["S"].items():
        #     if t == t_bar:
        #         print(f"S[{i}, {t_bar}, {s}] = {val}")

        return sol

    def RH_terminal_model(self, sol):
        """
        Builds and returns the deterministic terminal-stage model for the rolling horizon framework,
        covering the period from t_L to T after hurricane landfall.
        """

        V = self.network_processor.V
        V0 = self.network_processor.V0
        V1 = self.network_processor.V1
        V3 = self.network_processor.V3
        R = self.network_processor.R
        T = self.network_processor.T
        A = self.network_processor.A
        K_ijt = self.network_processor.K_ijt
        w_r = self.network_processor.w_r
        I_ir0 = self.network_processor.I_ir0
        g_shelter = self.network_processor.g_shelter
        g_pod = self.network_processor.g_pod
        t_ij = self.network_processor.t_ij  
        P = self.demand_processor.P
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij
        sce = range(self.num_sce)
        t_L = self.network_processor.t_L

        # Create the model
        model = gp.Model("Terminal_Model")

        I = {}
        
        for i in range(V):
            for r in range(R):
                for t in range(t_L, T):
                    I[i, r, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{r}_{t}")
        
        U = {}
        
        for i in range(V):
            for t in range(t_L, T):
                U[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_{i}_{t}")
        
        S = {}
        
        for i in range(V):
            for t in range(t_L, T):
                S[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{i}_{t}")
        
        f = {} 
        
        for i in range(V):
            for j in range(V):
                if (i, j) in A:
                    for r in range(R):
                        for t in range(t_L, T):
                            f[i, j, r, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}_{r}_{t}")

        # Constraints
        # # Initial conditions from sol
        # for i in range(V):
        #     for r in range(R):
        #         model.addConstr(I[i, r, t_L] == sol["I"][i, r, t_L - 1], name=f"InitInv_Terminal_{i}_{r}_{t_L}")

        for i in range(V):
            model.addConstr(U[i, t_L] == sol["U"][i, t_L - 1] + self.demand_processor.d_its[(i, t_L)] - S[i, t_L], name=f"InitUnmet_Terminal_{i}_{t_L}")

        # Flow balance only for t_L
        for i in range(V):
            for r in range(R):
                outgoing_flow = gp.quicksum(f[i, j, r, t_L] for j in range(V) if (i, j) in A)

                incoming_flow_regular = gp.quicksum(f[j, i, r, t_prime] for j in range(V) for t_prime in range(t_L, T) 
                                                    if (j, i) in A and t_prime == t_L - (t_ij[j][i] + 1))
                
                incoming_flow_fixed = gp.quicksum(self.fixed_flow[(j, i, r, t_prime)] for j in range(V) for t_prime in range(t_L) 
                                                    if (j, i) in A and t_prime == t_L - (t_ij[j][i] + 1))

                total_incoming = incoming_flow_regular + incoming_flow_fixed

                # Inventory update constraint for shelters
                if i in V1:
                    model.addConstr(I[i, r, t_L] == sol["I"][i, r, t_L - 1] - outgoing_flow + total_incoming - S[i, t_L] * g_shelter[r], 
                                    name=f"InventoryUpdate_Shelter_{i}_{r}_{t}")
                # Inventory update constraint for pods
                elif i in V3:
                    model.addConstr(I[i, r, t_L] == sol["I"][i, r, t_L - 1] - outgoing_flow + total_incoming - S[i, t_L] * g_pod[r],
                                    name=f"InventoryUpdate_Pod_{i}_{r}_{t}")
                # Inventory update constraint for warehouses and RSAs
                else:
                    model.addConstr(I[i, r, t_L] == sol["I"][i, r, t_L - 1] - outgoing_flow + total_incoming,
                                    name=f"InventoryUpdate_{i}_{r}_{t}")
            
        # Flow balance from t_L+1 to T
        for i in range(V):
            for r in range(R):
                for t in range(t_L + 1, T):
                    outgoing_flow = gp.quicksum(f[i, j, r, t] for j in range(V) if (i, j) in A)

                    incoming_flow_regular = gp.quicksum(f[j, i, r, t_prime] for j in range(V) for t_prime in range(t_L, T) 
                                                        if (j, i) in A and t_prime == t - (t_ij[j][i] + 1))
                    
                    incoming_flow_fixed = gp.quicksum(self.fixed_flow[(j, i, r, t_prime)] for j in range(V) for t_prime in range(t_L) 
                                                      if (j, i) in A and t_prime == t - (t_ij[j][i] + 1))

                    total_incoming = incoming_flow_regular + incoming_flow_fixed

                    # Inventory update constraint for shelters
                    if i in V1:
                        model.addConstr(I[i, r, t] == I[i, r, t-1] - outgoing_flow + total_incoming - S[i, t] * g_shelter[r], 
                                        name=f"InventoryUpdate_Shelter_{i}_{r}_{t}")
                    # Inventory update constraint for pods
                    elif i in V3:
                        model.addConstr(I[i, r, t] == I[i, r, t-1] - outgoing_flow + total_incoming - S[i, t] * g_pod[r],
                                        name=f"InventoryUpdate_Pod_{i}_{r}_{t}")
                    # Inventory update constraint for warehouses and RSAs
                    else:
                        model.addConstr(I[i, r, t] == I[i, r, t-1] - outgoing_flow + total_incoming,
                                        name=f"InventoryUpdate_{i}_{r}_{t}")

        for i in range(V):
            for t in range(t_L + 1, T): # Since the demand is the same across all scenarios for terminal stage, we just use the first scenario as reference
                model.addConstr(U[i, t] == U[i, t - 1] - S[i, t] + self.demand_processor.d_its[(i, t, 0)], name=f"DemandShortageUpdate_Terminal_{i}_{t}")

        for (i, j) in A:
            for t in range(t_L, T):
                model.addConstr(gp.quicksum(w_r[r] * f[i, j, r, t] for r in range(R)) <= K_ijt[i][j][t], name=f"FlowCapacity_Terminal_{i}_{j}_{t}")

        # Objective function
        objective = (gp.quicksum(p_i[i] * U[i, t] for i in range(V) for t in range(t_L, T)) + 
                     gp.quicksum(cost_ir[i][r] * f[i, j, r, t] for i in V0 for j in range(V) if (i, j) in A for r in range(R) for t in range(t_L, T)) + 
                     gp.quicksum(m_ij[i][j] * f[i, j, r, t] for i in range(V) for j in range(V) if (i, j) in A for r in range(R) for t in range(t_L, T)))

        model.setObjective(objective, GRB.MINIMIZE)
        model.setParam("OutputFlag", 0)
        model.update()

        return model, I, U, S, f
    
    def calculate_cost_rh(self, variable_values, t_bar):
        """
        Returns total cost (penalty + transportation) at time t_bar.
        """
        V = self.network_processor.V
        V0 = self.network_processor.V0
        R = self.network_processor.R
        A = self.network_processor.A
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij
        P = self.demand_processor.P
        sce = range(self.num_sce)  # Number of scenarios

        # Penalty cost for unmet demand
        penalty_cost = sum(P[s] * sum(p_i[i] * variable_values["U"][(i, t_bar, s)] for i in range(V)) for s in sce)

        # Procurement cost
        proc_cost = sum(P[s] * sum(cost_ir[i][r] * variable_values["f"][(i, j, r, t_bar, s)] for i in V0 for j in range(V) for r in range(R) if (i, j) in A) for s in sce)

        # Transportation cost
        transport_cost = sum(P[s] * sum(m_ij[i][j] * variable_values["f"][(i, j, r, t_bar, s)] for i in range(V) for j in range(V) for r in range(R) if (i, j) in A) for s in sce)

        # total = penalty_cost + proc_cost + transport_cost

        # print(f"Cost at t_bar={t_bar} = {total:.2f}")

        return penalty_cost, proc_cost, transport_cost

    def calculate_cost_terminal(self, variable_values, t_L):
        """
        Calculates the total cost (penalty + transportation) for the deterministic terminal model
        over time steps t ∈ {t_L + 1, ..., T - 1}.
        """
        V = self.network_processor.V
        V0 = self.network_processor.V0
        R = self.network_processor.R
        A = self.network_processor.A
        T = self.network_processor.T
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij

        # Penalty cost for unmet demand
        penalty_cost = sum(p_i[i] * variable_values["U"].get((i, t), 0) for i in range(V) for t in range(t_L + 1, T))

        # Transportation cost from source nodes
        transport_cost = sum(
            cost_ir[i][r] * variable_values["f"].get((i, j, r, t), 0) for i in V0 for j in range(V) if (i, j) in A for r in range(R) for t in range(t_L + 1, T))

        # Per-mile transportation cost
        transport_cost += sum(
            m_ij[i][j] * variable_values["f"].get((i, j, r, t), 0) for i in range(V) for j in range(V) if (i, j) in A for r in range(R) for t in range(t_L + 1, T))

        return penalty_cost + transport_cost

    def RH_evaluation(self, sample_paths, demand_processor, network_processor):
        """
        Evaluates the rolling horizon model over multiple out-of-sample sample paths.
        For each sample path:
        - Iteratively builds and solves the RH model at each time step,
        - Updates the state with realized values,
        - Tracks costs (penalty and transportation),
        - Solves a terminal model at the end of the horizon.

        Returns the average cost and the individual total costs for all sample paths.
        """
        start_time = time.time()

        # from demand_gen_indep import new_in_sample

        nbOS = demand_processor.nbOS
        T = network_processor.T
        V = network_processor.V
        t_L = network_processor.t_L
        V0 = self.network_processor.V0
        R = self.network_processor.R
        A = self.network_processor.A
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij

        # Ensure PoD location info is loaded (used for post-landfall PoD demand)
        if not hasattr(demand_processor, "pod_info_df"):
            demand_processor.pod_info_df = pd.read_excel("PoD location information.xlsx", sheet_name="Sheet1")

        # Load sample paths from CSV
        OS_paths_df = pd.read_csv(sample_paths)
        OS_paths = OS_paths_df.values
        total_costs = np.zeros(nbOS)  # Total cost for each sample path
        penalty_costs = np.zeros(nbOS) # Penalty cost for each sample path
        proc_costs = np.zeros(nbOS) # Procurement cost for each sample path
        transport_costs = np.zeros(nbOS) # Transportation cost for each sample path

        for k in range(nbOS):
            # print("Sample path=", k)
            sample_path_row = OS_paths[k]  # Example: sample_path_row[t_bar] gives the hurricane's realized state at t_bar
            self.fixed_flow = {}
            t_bar = 0
            sol = {}  # Initial state
            cost_k = 0  # Total cost for sample path k
            # Different types of cost for sample path k
            penalty_k = 0
            proc_k = 0
            transport_k = 0

            # Rolling loop over time steps
            while t_bar <= t_L:
                # print("t_bar=", t_bar)
                demand_processor.d_its = {} # Clear demand generated in the previous roll.
                s_t = eval(sample_path_row[t_bar])  # Converts '(-23.55, 0)' to (-23.55, 0)
                # print("s_t:", s_t)

                # Generate the demand at t_bar only based on the realized current state of the hurricane
                demand_processor.generate_current_demand(t_bar, s_t, network_processor, solution_method=4)

                # Generate multiple in-sample scenario paths starting from the current time step t_bar and realized hurricane state s_t.
                # Each scenario is a full trajectory of future hurricane states (track error, intensity) used to simulate uncertainty from t_bar onward.
                if self.num_sce_mean is not None: # Deterministic mean value RH
                    in_sample_paths = new_in_sample(t_bar, s_t, self.num_sce_mean)
                else:
                    in_sample_paths = new_in_sample(t_bar, s_t, self.num_sce)
                
                original_num_sce = self.demand_processor.num_sce
                # print("original num sce:", original_num_sce)
                if self.num_sce_mean is not None:
                    self.demand_processor.num_sce = self.num_sce_mean
                    # print("temporary num sce:", self.demand_processor.num_sce)
                
                # Generate the demand at t_bar + 1 to t_L based on the realized state of the hurricane at t_bar
                demand_processor.generate_future_demand(t_bar, t_L, network_processor, in_sample_paths, solution_method=4)  

                demand_processor.generate_post_landfall_demand(t_L, in_sample_paths, network_processor, solution_method=4)

                # Deterministic mean value RH
                self.demand_processor.num_sce = original_num_sce
                # print("back to original:", self.demand_processor.num_sce)

                # Collapse future scenario demand to mean (keep current t_bar demand as is)
                if self.num_sce_mean is not None:
                    # Average for t = t_bar+1 ... T-1, for each i
                    for t in range(t_bar + 1, T):
                        for i in range(V):
                            vals = [demand_processor.d_its[(i, t, s)] for s in range(self.num_sce_mean)]
                            demand_processor.d_its[(i, t, 0)] = float(np.mean(vals))
                
                # if self.num_sce_mean is not None:
                #     print("\n--- Demand used in model ---")
                #     for t in range(t_bar, T):
                #         print(f"\nt = {t}")
                #         for i in range(network_processor.V):
                #             if t == t_bar:
                #                 val = demand_processor.d_its[(i, t)]
                #             else:
                #                 val = demand_processor.d_its[(i, t, 0)]
                #             print(f"  i={i}, d = {val}")
                # exit(0)

                if t_bar < t_L:
                    # Build the model for current roll
                    model, I, U, S, f = self.RH_model_creator(t_bar, sol)
                    
                    # Solve the model
                    obj_val, variable_values = self.RH_model_solver(model, {"I": I, "U": U, "S": S, "f": f})           

                    penalty_pre, proc_pre, trans_pre = self.calculate_cost_rh(variable_values, t_bar)  
                    penalty_k += penalty_pre
                    proc_k += proc_pre
                    transport_k += trans_pre 
                    
                    cost_k += penalty_pre + proc_pre + trans_pre

                    # Save fixed flows (departed up to and including current t_bar) for the realized scenario only
                    for (i, j, r, t, s), val in variable_values["f"].items():
                        if t == t_bar:
                            if abs(val) < 1e-6:
                                self.fixed_flow[(i, j, r, t)] = 0
                            else:    
                                self.fixed_flow[(i, j, r, t)] = val
                
                    # Update the rolling state for next time step
                    sol = self.RH_sol_creator(variable_values, t_bar)

                    # Advance to the next roll
                    t_bar += 1
                
                else:
                    # Create the terminal deterministic model from t_L to T
                    terminal_model, I_term, U_term, S_term, f_term = self.RH_terminal_model(sol)

                    # Solve the terminal deterministic model from t_L to T
                    obj_term, var_term = self.RH_model_solver(terminal_model, {"I": I_term, "U": U_term, "S": S_term, "f": f_term})
                    
                    penalty_post = sum(p_i[i] * var_term["U"][(i, t)] for i in range(V) for t in range(t_L, T))
                    proc_post = sum(cost_ir[i][r] * var_term["f"][(i, j, r, t)] for i in V0 for j in range(V) if (i, j) in A for r in range(R) for t in range(t_L, T))
                    trans_post = sum(m_ij[i][j] * var_term["f"][(i, j, r, t)] for i in range(V) for j in range(V) if (i, j) in A for r in range(R) for t in range(t_L, T))

                    penalty_k += penalty_post
                    proc_k += proc_post
                    transport_k += trans_post

                    cost_k += penalty_post + proc_post + trans_post

                    t_bar += 1

            # Save the cost for this sample path
            total_costs[k] = cost_k
            penalty_costs[k] = penalty_k
            proc_costs[k] = proc_k
            transport_costs[k] = transport_k
            # Print the total cost for this path
            print(f"Total cost for sample path {k} = {cost_k:.2f}")

        avg_penalty = np.mean(penalty_costs)
        avg_proc = np.mean(proc_costs)
        avg_transport = np.mean(transport_costs)
        
        mean_cost = np.mean(total_costs)
        std_cost = np.std(total_costs, ddof=1)
        ci_low = mean_cost - 1.96 * std_cost / np.sqrt(nbOS)
        ci_high = mean_cost + 1.96 * std_cost / np.sqrt(nbOS)
        CI = mean_cost - ci_low
        print(f"\nRH Evaluation Results")
        print(f"μ ± 1.96*σ/√NS = {mean_cost:.2f} ± {CI:.2f}")
        print(f"95% Confidence Interval = [{ci_low:.2f}, {ci_high:.2f}]")

        print(f"\nAverage Cost Breakdown:")
        print(f"Average Penalty Cost = {avg_penalty:.2f}")
        print(f"Average Procurement Cost = {avg_proc:.2f}")
        print(f"Average Transportation Cost = {avg_transport:.2f}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total Execution Time for RH Evaluation: {total_time:.2f} seconds")

        return mean_cost, total_costs, total_time
    
    '''
    def RH_evaluation_new(self, sample_paths, demand_processor, network_processor):
        """
        Evaluates the rolling horizon model over multiple out-of-sample sample paths.
        For each sample path:
        - Iteratively builds and solves the RH model at each time step,
        - Updates the state with realized values,
        - Tracks costs (penalty and transportation),
        - Solves a terminal model at the end of the horizon.

        Returns the average cost and the individual total costs for all sample paths.
        """
        # from demand_gen_indep import new_in_sample

        T = network_processor.T
        P = self.demand_processor.P
        V = self.network_processor.V
        V0 = self.network_processor.V0
        R = self.network_processor.R
        A = self.network_processor.A
        nbOS = demand_processor.nbOS
        t_L = network_processor.t_L

        start_time = time.time()

        # Ensure PoD location info is loaded (used for post-landfall PoD demand)
        if not hasattr(demand_processor, "pod_info_df"):
            demand_processor.pod_info_df = pd.read_excel("PoD location information.xlsx", sheet_name="Sheet1")

        # Load sample paths from CSV
        OS_paths_df = pd.read_csv(sample_paths)
        OS_paths = OS_paths_df.values
        total_costs = np.zeros(nbOS)  # Total cost for each sample path

        # Stores models built during the first sample path
        # These models are reused across all sample paths to reduce memory usage
        model_dict = {} 

        for k in range(nbOS):
            print("Sample path=", k)
            sample_path_row = OS_paths[k]  # 1D array for path k
            self.fixed_flow = {}
            t_bar = 0
            sol = {}  # Initial state
            cost_k = 0  # Total cost for sample path k

            # Rolling loop over time steps
            while t_bar <= t_L:
                print("t_bar=", t_bar)
                s_t = eval(sample_path_row[t_bar])  # Converts '(-23.55, 0)' to (-23.55, 0)

                # Generate the demand at t_bar only based on the realized current state of the hurricane
                demand_processor.generate_current_demand(t_bar, s_t, network_processor, solution_method=4)

                # Generate multiple in-sample scenario paths starting from the current time step t_bar and realized hurricane state s_t.
                # Each scenario is a full trajectory of future hurricane states (track error, intensity) used to simulate uncertainty from t_bar onward.
                in_sample_paths = new_in_sample(t_bar, s_t, self.num_sce)

                # Identify which in-sample scenario matches the realized state s_t
                s_realized = None
                for s in range(self.num_sce):
                    if tuple(in_sample_paths[s, t_bar]) == s_t:
                        s_realized = s
                        break

                if s_realized is None:
                    raise ValueError(f"No in-sample scenario matches realized state s_t = {s_t} at t_bar = {t_bar}")

                # Generate the demand at t_bar + 1 to t_L based on the realized state of the hurricane at t_bar
                demand_processor.generate_future_demand(t_bar, s_t, t_L, network_processor, in_sample_paths, solution_method=4)  

                demand_processor.generate_post_landfall_demand(t_L, in_sample_paths, network_processor, solution_method=4)

                # Build the model for current roll (first sample path only)
                if k == 0:
                    model, I, U, S, f = self.RH_model_creator(t_bar, sol)
                    model_dict[t_bar] = (model, I, U, S, f)

                else:
                    model, I, U, S, f = model_dict[t_bar]

                    ### Update the new model info ###
                    # Only update model state if t_bar > 0 (sol is available)
                    if t_bar > 0:
                        self.update_init_data(model, sol, t_bar)
                    self.update_demand_data(model, demand_processor, t_bar)


                # Solve the model
                obj_val, variable_values = self.RH_model_solver(model, {"I": I, "U": U, "S": S, "f": f})
                print(f"Objective value at time t_bar = {t_bar}: {obj_val}")

                if obj_val is None:
                    print(f"Skipping sample path {k} at t_bar={t_bar} due to infeasibility.")
                    exit(0)              

                cost_k += self.calculate_cost_rh(variable_values, t_bar, s_t)

                # Accumulate fixed flows made at t_bar
                for (i, j, r, t, s), val in variable_values["f"].items():
                    if t == t_bar:
                        self.fixed_flow[(i, j, r, t, s)] = val
                
                # Update the rolling state for next time step (should be before model creation, because we need updated sol to create the model every time)
                sol = self.RH_sol_creator(variable_values, t_bar, s_realized)

                # Advance to the next time step
                t_bar += 1
                
            # Now solve the terminal deterministic model from t_L + 1 to T - 1
            terminal_model, I_term, U_term, S_term, f_term = self.RH_terminal_model(sol, s_t, sample_paths)

            obj_term, var_term = self.RH_model_solver(terminal_model, {"I": I_term, "U": U_term, "S": S_term, "f": f_term})
            print(f"Objective value from terminal model (t ∈ [{t_L + 1}, {T - 1}]): {obj_term}")

            if obj_term is not None:
                cost_k += self.calculate_cost_terminal(var_term, t_L)
            else:
                print(f"Terminal stage for sample path {k} was infeasible.")
                exit(0)   

            # Save the cost for this sample path
            total_costs[k] = cost_k

        # Average cost across all sample paths
        average_cost = np.mean(total_costs)

        return average_cost, total_costs

    def update_demand_data(self, model, demand_processor, t_bar):
        """
        Updates RHS for demand shortage update constraints at each t >= t_bar.
        Handles both scenario-based (i, t, s) and deterministic post-landfall (i, t) keys.
        """
        for key, demand_val in demand_processor.d_its.items():
            if len(key) == 3:
                i, t, s = key
                if t >= t_bar:
                    cname = f"DemandShortageUpdate_{i}_{t}_{s}"
                    constr = model.getConstrByName(cname)
                    if constr is not None:
                        constr.setAttr("RHS", demand_val)

            elif len(key) == 2:
                i, t = key
                if t >= t_bar:
                    # Update this for all scenarios, since post-landfall demand is same for all s
                    for s in range(self.num_sce):  # Or use network_processor.num_sce
                        cname = f"DemandShortageUpdate_{i}_{t}_{s}"
                        constr = model.getConstrByName(cname)
                        if constr is not None:
                            constr.setAttr("RHS", demand_val)

        model.update()

    def update_init_data(self, model, sol, t_bar):
        """
        Updates the reused model at time t_bar > 0 with the new sol dictionary.
        This updates:
        - Initial inventory constraints: I[i, r, t_bar] == sol["I"][i, r, t_bar]
        - Initial unmet demand constraints: U[i, t_bar, s] == sol["U"][i, t_bar]
        """
        # Update initial inventory values
        for (i, r, t), value in sol["I"].items():
            if t == t_bar:
                cname = f"InitInv_{i}_{r}_{t_bar}"
                constr = model.getConstrByName(cname)
                if constr is not None:
                    constr.setAttr("RHS", value)

        # Update initial unmet demand values
        for (i, t), value in sol["U"].items():
            if t == t_bar:
                for s in range(self.num_sce):
                    cname = f"InitUnmet_{i}_{s}_{t_bar}"
                    constr = model.getConstrByName(cname)
                    if constr is not None:
                        constr.setAttr("RHS", value)

        model.update()

    def RH_evaluation_parallel(self, sample_paths_file, demand_processor, network_processor):
        """
        Parallel evaluation of the rolling horizon model over multiple out-of-sample sample paths.
        Each path is solved independently in a separate process.
        Returns the average cost and individual total costs.
        """
        # from demand_gen_indep import new_in_sample

        # Load all sample paths from CSV
        OS_paths_df = pd.read_csv(sample_paths_file)

        # Use only as many OOS paths as specified by demand_processor.nbOS
        nbOS = demand_processor.nbOS
        OS_paths = OS_paths_df.values[:nbOS]

        print(f"Starting parallel RH evaluation for {nbOS} sample paths...")

        total_costs = [None] * nbOS  # to preserve order

        # Run in parallel using available CPU cores
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.solve_single_sample_path,
                    k,
                    OS_paths[k],
                    sample_paths_file,
                    demand_processor,
                    network_processor
                )
                for k in range(nbOS)
            ]

            for k, future in enumerate(futures):
                try:
                    total_costs[k] = future.result()
                    print(f"Sample path {k} completed with cost {total_costs[k]:.2f}")
                except Exception as e:
                    print(f"Sample path {k} failed: {e}")
                    total_costs[k] = np.nan

        average_cost = np.nanmean(total_costs)

        print(f"\n Average Rolling Horizon Cost (across {nbOS} OOS paths): {average_cost:.2f}")

        return average_cost, total_costs

    def solve_single_sample_path(self, k, sample_path_row, sample_paths, demand_processor, network_processor):
        """
        Solves a single out-of-sample sample path (k).
        Used by the parallel RH evaluation.
        """
        # from demand_gen_indep import new_in_sample

        t_L = network_processor.t_L
        T = network_processor.T

        print("Sample path=", k)
        self.fixed_flow = {}
        t_bar = 0
        sol = {}  # Initial state
        cost_k = 0  # Total cost for sample path k

        # Rolling loop over time steps
        while t_bar <= t_L:
            print("t_bar=", t_bar)
            s_t = eval(sample_path_row[t_bar])  # Converts '(-23.55, 0)' to (-23.55, 0)

            # Generate the demand at t_bar only based on the realized current state of the hurricane
            demand_processor.generate_current_demand(t_bar, s_t, network_processor, solution_method=4)

            # Generate multiple in-sample scenario paths starting from the current time step t_bar and realized hurricane state s_t.
            # Each scenario is a full trajectory of future hurricane states (track error, intensity) used to simulate uncertainty from t_bar onward.
            in_sample_paths = new_in_sample(t_bar, s_t, self.num_sce)

            # Identify which in-sample scenario matches the realized state s_t
            s_realized = None
            for s in range(self.num_sce):
                if tuple(in_sample_paths[s, t_bar]) == s_t:
                    s_realized = s
                    break

            if s_realized is None:
                raise ValueError(f"No in-sample scenario matches realized state s_t = {s_t} at t_bar = {t_bar}")

            # Generate the demand at t_bar + 1 to t_L based on the realized state of the hurricane at t_bar
            demand_processor.generate_future_demand(t_bar, s_t, t_L, network_processor, in_sample_paths, solution_method=4)  

            demand_processor.generate_post_landfall_demand(t_L, in_sample_paths, network_processor, solution_method=4)

            # Build the model for current roll
            model, I, U, S, f = self.RH_model_creator(t_bar, sol)
            
            # Solve the model
            obj_val, variable_values = self.RH_model_solver(model, {"I": I, "U": U, "S": S, "f": f})
            print(f"Objective value at time t_bar = {t_bar}: {obj_val}")

            if obj_val is None:
                print(f"Skipping sample path {k} at t_bar={t_bar} due to infeasibility.")
                exit(0)              

            cost_k += self.calculate_cost_rh(variable_values, t_bar, s_t)

            # Accumulate fixed flows made at t_bar
            for (i, j, r, t, s), val in variable_values["f"].items():
                if t == t_bar:
                    self.fixed_flow[(i, j, r, t, s)] = val
            
            # Update the rolling state for next time step (should be before model creation, because we need updated sol to create the model every time)
            sol = self.RH_sol_creator(variable_values, t_bar, s_realized)

            # Advance to the next time step
            t_bar += 1
            
        # Now solve the terminal deterministic model from t_L + 1 to T - 1
        terminal_model, I_term, U_term, S_term, f_term = self.RH_terminal_model(sol, s_t, sample_paths)

        obj_term, var_term = self.RH_model_solver(terminal_model, {"I": I_term, "U": U_term, "S": S_term, "f": f_term})
        print(f"Objective value from terminal model (t ∈ [{t_L + 1}, {T - 1}]): {obj_term}")

        if obj_term is not None:
            cost_k += self.calculate_cost_terminal(var_term, t_L)
        else:
            print(f"Terminal stage for sample path {k} was infeasible.")
            exit(0)   

        # Return the cost for this sample path
        return cost_k
    '''

    ### First-stage
    def master_problem(self, thetaLB, model_version=1):

        V = self.network_processor.V
        R = self.network_processor.R
        T = self.network_processor.T
        V0 = self.network_processor.V0
        V3 = self.network_processor.V3
        T_Bef = self.network_processor.T_Bef
        A = self.network_processor.A
        I_ir0 = self.network_processor.I_ir0
        C_i0 = self.network_processor.C_i0
        h_i = self.network_processor.h_i
        t_ij = self.network_processor.t_ij
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

        # Add vehicle-related first-stage decision variables for Model 2
        if model_version == 2:
            self.K = {}
            for i in range(V):
                for j in range(V):
                    if (i, j) in A:
                        for t in range(T):
                            self.K[i, j, t] = self.master_model.addVar(vtype=GRB.INTEGER, lb=0, name=f"K_{i}_{j}_{t}")

            self.J = {}
            for i in range(V):
                for t in range(T):
                    self.J[i, t] = self.master_model.addVar(vtype=GRB.INTEGER, lb=0, name=f"J_{i}_{t}")

            # Vehicle initializtion constraint
            for i in range(V):
                self.master_model.addConstr(self.J[i, 0] == C_i0[i], name="VehicleInit_%s" % i)
            
            # Vehicle flow balance constraint
            for i in range(V):
                for t in range(T - 1):
                    outgoing = gp.quicksum(self.K[i, j, t] for j in range(V) if (i, j) in A)
                    incoming = gp.LinExpr()
                    for j in range(V):
                        if (j, i) in A:
                            t_prime = t - t_ij[j][i] - 1 - h_i[i]
                            if t_prime >= 0:
                                incoming += self.K[j, i, t_prime]
                    self.master_model.addConstr(self.J[i, t + 1] == self.J[i, t] - outgoing + incoming, name="VehicleCont_%s_%s" % (i, t))

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

        if model_version == 1:
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

    def optimality_sub(self, model_version=1):
        V = self.network_processor.V
        R = self.network_processor.R
        T = self.network_processor.T
        A = self.network_processor.A
        V0 = self.network_processor.V0
        V1 = self.network_processor.V1
        V3 = self.network_processor.V3
        t_ij = self.network_processor.t_ij
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
        for (i, j) in A:
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
        self.pi_opt_init_flow = {}
        for i in range(V):
            for r in range(R):
                # Add the constraint to the model
                self.pi_opt_init_flow[i, r, 0] = self.sub_opt.addConstr(
                    -gp.quicksum(self.f_opt[i, j, r, 0] for j in range(V) if (i, j) in A)
                    - self.S_opt[i, 0] * (g_shelter[r] if i in V1 else g_pod[r] if i in V3 else 0) == 0,
                    name=f"flow_conservation_opt_{i}_{r}_{t}")

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
                for t in range(1, T):
                    # Add the constraint to the model
                    self.pi_opt_flow_conservation[i, r, t] = self.sub_opt.addConstr(
                        -gp.quicksum(self.f_opt[i, j, r, t] for j in range(V) if (i, j) in A)
                        + gp.quicksum(self.f_opt[j, i, r, t - (t_ij[j][i] + 1)] for j in range(V) if (j, i) in A and t - (t_ij[j][i] + 1) >= 0)
                        - self.S_opt[i, t] * (g_shelter[r] if i in V1 else g_pod[r] if i in V3 else 0) == 0,
                        name=f"flow_conservation_opt_{i}_{r}_{t}")
        
        # Arc capacity constraint
        self.pi_opt_arc_capacity = {}
        for (i, j) in A:  # Only consider valid links in A
            for t in range(T):
                if model_version == 2:
                    self.pi_opt_arc_capacity[i, j, t] = self.sub_opt.addConstr(-gp.quicksum(w_r[r] * self.f_opt[i, j, r, t] for r in range(R)) >= 0,
                        name=f"arc_capacity_opt_{i}_{j}_{t}")
                else:
                    self.pi_opt_arc_capacity[i, j, t] = self.sub_opt.addConstr(-gp.quicksum(w_r[r] * self.f_opt[i, j, r, t] for r in range(R)) >= -K_ijt[i][j][t],
                        name=f"arc_capacity_opt_{i}_{j}_{t}")

        self.sub_opt.setParam("OutputFlag", 0)

    def feasibility_sub(self, model_version=1):
        V = self.network_processor.V
        R = self.network_processor.R
        T = self.network_processor.T
        A = self.network_processor.A
        V1 = self.network_processor.V1
        V3 = self.network_processor.V3
        t_ij = self.network_processor.t_ij
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
                        + gp.quicksum(self.f_feas[j, i, r, t - t_ij[j][i] + 1] for j in range(V) if (i, j) in A and t - t_ij[j][i] + 1 >= 0)
                        - self.S_feas[i, t] * (g_shelter[r] if i in V1 else g_pod[r] if i in V3 else 0)
                        + self.epsilon_1[i, r, t] - self.epsilon_2[i, r, t] == 0,
                        name=f"flow_conservation_feas_{i}_{r}_{t}")
        
        # Flow capacity constraint for all links, time periods, and scenarios
        self.pi_feas_arc_capacity = {}
        for i in range(V):
            for j in range(V):
                if (i, j) in A:  # Only consider valid links in A
                    for t in range(T):
                        if model_version == 2:  
                            self.pi_feas_arc_capacity[i, j, t] = self.sub_feas.addConstr(
                                -gp.quicksum(w_r[r] * self.f_feas[i, j, r, t] for r in range(R))
                                + self.epsilon_3[i, j, t] >= 0,
                                name=f"arc_capacity_feas_{i}_{j}_{t}")
                        else:
                            self.pi_feas_arc_capacity[i, j, t] = self.sub_feas.addConstr(
                                -gp.quicksum(w_r[r] * self.f_feas[i, j, r, t] for r in range(R))
                                + self.epsilon_3[i, j, t] >= -K_ijt[i][j][t],
                                name=f"arc_capacity_feas_{i}_{j}_{t}")

        # Disable Gurobi solver output
        self.sub_feas.setParam("OutputFlag", 0)

    def benders(self, eps, thetaLB, time_limit, model_version=1):

        V = self.network_processor.V
        R = self.network_processor.R
        T = self.network_processor.T
        A = self.network_processor.A
        V0 = self.network_processor.V0
        V1 = self.network_processor.V1
        V3 = self.network_processor.V3

        t_ij = self.network_processor.t_ij
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij
        g_shelter = self.network_processor.g_shelter
        g_pod = self.network_processor.g_pod
        w_r = self.network_processor.w_r
        K_ijt = self.network_processor.K_ijt
        P = self.demand_processor.P
        sce = range(len(P))

        start_time = time.time()
        # Initialize bounds and tolerance
        self.LB = thetaLB
        self.UB = 1e10

        # Counters for optimality and feasibility cuts
        self.Ocuts = 0
        self.Fcuts = 0

        # Iteration counter
        self.iter = 0
        
        # Number of scenarios
        num = len(sce)

        # Store first-stage decision variable values
        self.I_values = {}        
        self.theta_value = 0
        if model_version == 2:
            self.K_values = {}

        # Storage for optimality subproblem dual multipliers
        self.pi_opt_inventory_balance_U_val = {s: {} for s in range(num)}
        self.pi_opt_initial_shortage_U_val = {s: {} for s in range(num)}
        self.pi_opt_flow_conservation_val = {s: {} for s in range(num)}
        self.pi_opt_arc_capacity_val = {s: {} for s in range(num)}
        
        # Storage for feasibility subproblem dual multipliers
        self.pi_feas_inventory_balance_U_val = {}
        self.pi_feas_initial_shortage_U_val = {}
        self.pi_feas_flow_conservation_val = {}
        self.pi_feas_arc_capacity_val = {}

        # Start the Benders loop
        while (self.UB - self.LB) / max(1e-10, abs(self.UB)) > eps:
            self.iter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                print(f"\nStopping Benders after {elapsed_time:.2f} seconds (Time limit reached)")
                break  # Stop execution if the time limit is reached

            # Solve the master problem
            self.master_model.optimize()

            # Check if the optimization was successful
            if self.master_model.status == GRB.OPTIMAL:
                # Extract the values of first-stage decision variables I[i, r, t]
                for i in range(V):
                    for r in range(R):
                        for t in range(T):
                            self.I_values[i, r, t] = self.I[i, r, t].x  # Store decision variable values

                # Extract K[i, j, t] only for Model 2
                if model_version == 2:
                    for i in range(V):
                        for j in range(V):
                            if (i, j) in A:
                                for t in range(T):
                                    self.K_values[i, j, t] = self.K[i, j, t].x

                # Get the value of theta
                self.theta_value = self.theta.x

                # Update the lower bound
                self.LB = self.master_model.objVal       
                
            elif self.master_model.status == GRB.INFEASIBLE:
                print("Master problem is infeasible.")
                exit(0)  # Or handle infeasibility appropriately

            elif self.master_model.status == GRB.UNBOUNDED:
                print("Master problem is unbounded.")
                exit(0)  # Or handle unboundedness appropriately

            else:
                print(f"Optimization was stopped with status {self.master_model.status}")
                exit(0)

            # Initialize storage for subproblem solutions
            self.Q_opt = np.zeros(num)  # Stores optimality subproblem objective values
            self.w_feas = np.zeros(num)  # Stores feasibility subproblem objective values
            self.FLAG = 0  # (0 if I_hat is feasible for all scenarios, 1 otherwise)
            self.infeasible_sce = -1  # Track which scenario is infeasible

            # Loop over all scenarios
            for s in range(num):
                # Update RHS for inventory flow balance constraints
                for i in range(V):
                    for t in range(1, T):
                        self.pi_opt_inventory_balance_U[i, t].setAttr(GRB.Attr.RHS, self.demand_processor.d_its[(i, t, s)])

                # Update RHS for initial shortage constraints
                for i in range(V):
                    self.pi_opt_initial_shortage_U[i].setAttr(GRB.Attr.RHS, self.demand_processor.d_its[(i, 0, s)])

                # Update RHS for flow conservation constraints
                for i in range(V):
                    for r in range(R):
                        for t in range(T - 1):
                            self.pi_opt_flow_conservation[i, r, t].setAttr(GRB.Attr.RHS, self.I_values[i, r, t + 1] - self.I_values[i, r, t])

                # Update RHS for arc capacity constraints only for Model 2
                if model_version == 2:
                    for i in range(V):
                        for j in range(V):
                            if (i, j) in A:
                                for t in range(T):
                                    self.pi_opt_arc_capacity[i, j, t].setAttr(GRB.Attr.RHS, -self.K_values[i, j, t])

                # Solve the optimality subproblem
                self.sub_opt.optimize()

                if self.sub_opt.status == GRB.OPTIMAL:
                    # Store the optimal objective value
                    self.Q_opt[s] = self.sub_opt.objVal

                    # Retrieve and store dual values for each constraint
                    for key, constraint in self.pi_opt_inventory_balance_U.items():
                        self.pi_opt_inventory_balance_U_val[s][key] = constraint.pi
                    for key, constraint in self.pi_opt_initial_shortage_U.items():
                        self.pi_opt_initial_shortage_U_val[s][key] = constraint.pi
                    for key, constraint in self.pi_opt_flow_conservation.items():
                        self.pi_opt_flow_conservation_val[s][key] = constraint.pi
                    for key, constraint in self.pi_opt_arc_capacity.items():
                        self.pi_opt_arc_capacity_val[s][key] = constraint.pi

                elif self.sub_opt.status == GRB.INFEASIBLE:
                    # If infeasible, set FLAG to 1 and mark the scenario
                    self.FLAG = 1
                    self.infeasible_sce = s
                    break;  # No need to continue checking other scenarios

                else:
                    print(f"Error: One of the scenario subproblems is infeasible (Status: {self.model.sub_opt.status})")
                    exit(0)

            print("FLAG =", self.FLAG)

            if self.FLAG == 1:
                # Solve feasibility subproblem for the detected infeasible scenario
                s = self.infeasible_sce

                # Update RHS for inventory flow balance constraints
                for i in range(V):
                    for t in range(1, T):
                        self.pi_feas_inventory_balance_U[i, t].setAttr(GRB.Attr.RHS, self.demand_processor.d_its[(i, t, s)])

                # Update RHS for initial shortage constraints
                for i in range(V):
                    self.pi_feas_initial_shortage_U[i].setAttr(GRB.Attr.RHS, self.demand_processor.d_its[(i, 0, s)])

                # Update RHS for flow conservation constraints
                for i in range(V):
                    for r in range(R):
                        for t in range(T - 1):
                            self.pi_feas_flow_conservation[i, r, t].setAttr(GRB.Attr.RHS, self.I_values[i, r, t + 1] - self.I_values[i, r, t])

                # Update RHS for arc capacity constraints only for Model 2
                if model_version == 2:
                    for i in range(V):
                        for j in range(V):
                            if (i, j) in A:
                                for t in range(T):
                                    self.pi_feas_arc_capacity[i, j, t].setAttr(GRB.Attr.RHS, -self.K_values[i, j, t])

                # Solve the feasibility subproblem
                self.sub_feas.optimize()

                # Store the objective value (should be positive if infeasible)
                self.w_feas[s] = self.sub_feas.objVal

                # Retrieve dual values from feasibility subproblem constraints
                for key, constraint in self.pi_feas_inventory_balance_U.items():
                    self.pi_feas_inventory_balance_U_val[key] = constraint.pi
                for key, constraint in self.pi_feas_initial_shortage_U.items():
                    self.pi_feas_initial_shortage_U_val[key] = constraint.pi
                for key, constraint in self.pi_feas_flow_conservation.items():
                    self.pi_feas_flow_conservation_val[key] = constraint.pi
                for key, constraint in self.pi_feas_arc_capacity.items():
                    self.pi_feas_arc_capacity_val[key] = constraint.pi

                # Generate feasibility cut if w_feas is large
                if self.w_feas[s] > eps:
                    feasibility_cut_expr = gp.LinExpr()

                    # Add inventory flow balance terms
                    for key, dual_val in self.pi_feas_inventory_balance_U_val.items():
                        i, t = key
                        feasibility_cut_expr += dual_val * self.demand_processor.d_its[(i, t, s)]

                    # Add initial shortage terms
                    for key, dual_val in self.pi_feas_initial_shortage_U_val.items():
                        i = key
                        feasibility_cut_expr += dual_val * self.demand_processor.d_its[(i, 0, s)]

                    # Add flow conservation terms
                    for key, dual_val in self.pi_feas_flow_conservation_val.items():
                        i, r, t = key
                        if abs(dual_val) > 1e-7:
                            feasibility_cut_expr += dual_val * (self.I[i, r, t + 1] - self.I[i, r, t])

                    # Add arc capacity terms
                    for key, dual_val in self.pi_feas_arc_capacity_val.items():
                        i, j, t = key
                        if model_version == 2:
                            feasibility_cut_expr += dual_val * (-self.K[i, j, t])
                        else:
                            feasibility_cut_expr += dual_val * -K_ijt[i][j][t]

                    # Add the feasibility cut to the master problem
                    self.master_model.addConstr(feasibility_cut_expr <= 0, name=f"feasibility_cut_scenario_{s}")
                    self.Fcuts += 1  # Increment feasibility cuts counter

            else:
                print("here")
                # Compute a temporary upper bound using second-stage costs
                self.tempUB = sum(self.Q_opt[s] * P[s] for s in range(num))

                # Update the upper bound
                if self.tempUB < self.UB:
                    self.UB = self.tempUB

                # Check convergence: If (UB - LB) is small, stop
                if (self.UB - self.LB) / max(1e-10, abs(self.LB)) < eps:
                    break

                else:
                    # Initialize the optimality cut expression
                    optimality_cut_expr = gp.LinExpr()
                    for s in range(num):
                        # Add inventory flow balance constraints
                        for key, dual_val in self.pi_opt_inventory_balance_U_val[s].items():
                            i, t = key
                            optimality_cut_expr += P[s] * dual_val * self.demand_processor.d_its[(i, t, s)]

                        # Add initial shortage constraints
                        for key, dual_val in self.pi_opt_initial_shortage_U_val[s].items():
                            i = key
                            optimality_cut_expr += P[s] * dual_val * self.demand_processor.d_its[(i, 0, s)]

                        # Add flow conservation constraints
                        for key, dual_val in self.pi_opt_flow_conservation_val[s].items():
                            i, r, t = key
                            if abs(dual_val) > 1e-7:
                                optimality_cut_expr += P[s] * dual_val * (self.I[i, r, t + 1] - self.I[i, r, t])

                        # Add arc capacity constraints
                        for key, dual_val in self.pi_opt_arc_capacity_val[s].items():
                            i, j, t = key
                            if model_version == 2:
                                optimality_cut_expr += P[s] * dual_val * (-self.K[i, j, t])
                            else:
                                optimality_cut_expr += P[s] * dual_val * -K_ijt[i][j][t]

                    # Add the optimality cut to the master model
                    self.master_model.addConstr(self.theta >= optimality_cut_expr, name="optimality_cut")
                    self.Ocuts += 1  # Increment the counter for optimality cuts

            # Print iteration details
            print(f"Iteration {self.iter}:")
            print(f"Lower Bound (LB) = {self.LB}")
            print(f"Upper Bound (UB) = {self.UB}")
            print("===========================================")

        # Final summary after convergence
        print("#########################################################################################")
        print(f"Final Iteration: {self.iter}")
        print(f"Number of Feasibility Cuts Added: {self.Fcuts}")
        print(f"Number of Optimality Cuts Added: {self.Ocuts}")
        print(f"Final Lower Bound (LB): {self.LB}")
        print(f"Final Upper Bound (UB): {self.UB}")
        print("#########################################################################################")

        # Return results
        return self.LB, self.UB, self.iter, self.Fcuts, self.Ocuts

    def compute_mvp_objective(self, I_mvp_values):

        P = self.demand_processor.P
        V = self.network_processor.V
        R = self.network_processor.R
        T = self.network_processor.T
        I_ir0 = self.network_processor.I_ir0
        d_its = self.demand_processor.d_its
        sce = range(len(P))
        mvpobj = 0  # Initialize the MVP objective value

        # print("Original stochastic demand inside mvp_objective (d_its):")
        # for key in sorted(d_its):
        #     val = d_its[key]
        #     if abs(val) >= 1e-6:
        #         print(f"d_its{key} = {val:.2f}")

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
                    for t in range(1, T):
                        self.pi_opt_flow_conservation[i, r, t].setAttr(GRB.Attr.RHS, I_mvp_values[i, r, t] - I_mvp_values[i, r, t-1])

            for i in range(V):
                for r in range(R):
                    self.pi_opt_init_flow[i, r, 0].setAttr(GRB.Attr.RHS, I_mvp_values[i, r, 0] - I_ir0[i][r])
            
            
            self.sub_opt.update()
            # Solve the optimality subproblem
            self.sub_opt.optimize()    
            # print("objective:", self.sub_opt.objVal)

            if self.sub_opt.status == GRB.OPTIMAL:
                # Update the optimal objective value for each scenario
                mvpobj += P[s] * self.sub_opt.objVal
            else:
                print(f"Model for scenario{s} did NOT solve to optimality. Status code = {self.sub_opt.status}")

        return mvpobj
