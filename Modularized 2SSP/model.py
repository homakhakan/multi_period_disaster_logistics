import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from demand_data import DemandData
import time

### Extended
class TwoStageSPModel:
    def __init__(self, network_processor, demand_processor, num_sce):
        self.network_processor = network_processor  # Store network data
        self.demand_processor = demand_processor  # Store demand data
        if not hasattr(self.demand_processor, 'd_its_original'):
            self.demand_processor.d_its_original = self.demand_processor.d_its
        if not hasattr(self.demand_processor, 'P_original'):
            self.demand_processor.P_original = self.demand_processor.P.copy()

        self.num_sce = num_sce  # Use full scenarios for other models

    def ext_model(self, is_mvp=False):

        if is_mvp:
            self.demand_processor.d_its = self.demand_processor.d_its_expected
            self.demand_processor.P = [1]
            self.num_sce = 1
        else:
            self.demand_processor.d_its = self.demand_processor.d_its_original
            self.demand_processor.P = self.demand_processor.P_original
            self.num_sce = len(self.demand_processor.P_original)

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
        P = self.demand_processor.P
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
    
    def RH_model_creator(self, t_bar, s_t, network_processor, demand_processor, sol, fixed_flow_all):

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
                    I[i, r, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="I_%s_%s_%s" % (i, r, t))
        
        # Define second-stage decision variables
        U = {}  # Demand shortage at location i at time period t in scenario s
        
        for i in range(V):
            for t in range(t_bar, T):
                for s in sce:  # Iterate directly over the scenario set (e.g., 0 and 1)
                    U[i, t, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="U_%s_%s_%s" % (i, t, s))
        
        S = {}  # Demand satisfied at location i at time period t in scenario s
        
        for i in range(V):
            for t in range(t_bar, T):
                for s in sce:  # Iterate directly over the scenario set
                    S[i, t, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S_%s_%s_%s" % (i, t, s))
        
        f = {}  # Quantity of commodity r traveling on link (i, j) at time period t in scenario s
        
        for i in range(V):
            for j in range(V):
                if (i, j) in A:  # Only define f_ijrt^s if (i, j) is a valid link
                    for r in range(R):
                        for t in range(t_bar, T):
                            for s in sce:  # Iterate directly over the scenario set
                                f[i, j, r, t, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_%s_%s_%s_%s_%s" % (i, j, r, t, s))

        if t_bar == 0:
            #Starting Value of I[t=0]
            for i in range(V):
                for r in range(R):
                    model.addConstr(I[i,r,0]==I_ir0[i][r])

            # Initial demand shortage for all locations and scenarios at time period t=0
            for i in range(V):
                for s in sce:
                    model.addConstr(U[i, 0, s] == self.demand_processor.d_its[(i, 0, s)], name="InitialDemandShortage_%s_%s" % (i, s))

            # Inventory update constraint for all locations, commodities, time periods (except last), and scenarios
            for i in range(V):
                for r in range(R):
                    for t in range(t_bar, T - 1):
                        for s in sce:
                            outgoing_flow = gp.quicksum(f[i, j, r, t, s] for j in range(V) if (i, j) in A)
                            incoming_flow = gp.quicksum(f[j, i, r, t - t_ijt[j][i] + 1, s] for j in range(V) if (j, i) in A and t - t_ijt[j][i] + 1 >= 0)
                            
                            # Inventory update constraint for shelters
                            if i in V1:
                                model.addConstr(
                                    I[i, r, t + 1] == I[i, r, t] - outgoing_flow + incoming_flow - S[i, t, s] * g_shelter[r],
                                    name="InventoryUpdate_Shelter_%s_%s_%s_%s" % (i, r, t, s)
                                )
                            # Inventory update constraint for pods
                            elif i in V3:
                                model.addConstr(
                                    I[i, r, t + 1] == I[i, r, t] - outgoing_flow + incoming_flow - S[i, t, s] * g_pod[r],
                                    name="InventoryUpdate_Pod_%s_%s_%s_%s" % (i, r, t, s)
                                )
                            # Inventory update constraint for warehouses and RSAs
                            else:
                                model.addConstr(
                                    I[i, r, t + 1] == I[i, r, t] - outgoing_flow + incoming_flow,
                                    name="InventoryUpdate_%s_%s_%s_%s" % (i, r, t, s))

        else:
            # Updated initial inventory
            for i in range(V):
                for r in range(R):
                    model.addConstr(
                        I[i, r, t_bar] == sol["I"][i, r, t_bar],
                        name=f"InitInv_{i}_{r}_{t_bar}")
                    
            # Updated initial unmet demand        
            for i in range(V):
                for s in sce:
                    model.addConstr(
                        U[i, t_bar, s] == sol["U"][i, t_bar],
                        name=f"InitUnmet_{i}_{s}_{t_bar}")
                    
            # Updated flow for all locations, commodities, time periods (except last), and scenarios
            for i in range(V):
                for r in range(R):
                    for t in range(t_bar, T - 1):
                        for s in sce:
                            outgoing_flow = gp.quicksum(f[i, j, r, t, s] for j in range(V) if (i, j) in A)
                            
                            # Regular incoming flow (from inside RH window)
                            incoming_flow_regular = gp.quicksum(
                                f[j, i, r, t - t_ijt[j][i] + 1, s]
                                for j in range(V)
                                if (j, i) in A and t - t_ijt[j][i] + 1 >= t_bar)
                            
                            # Fixed flow arriving now from earlier rolls
                            # incoming_flow_fixed = gp.quicksum(
                            #     sol["fixed_flow"].get((j, i, r, t, s), 0)
                            #     for j in range(V) if (j, i) in A and t - t_ijt[j][i] < t_bar) #***

                            incoming_flow_fixed = gp.quicksum(
                            fixed_flow_all.get((j, i, r, t_prime, s), 0)
                            for j in range(V)
                            for t_prime in range(t_bar)  # Flows made before t_bar
                            if (j, i) in A and t_prime + t_ijt[j][i] == t + 1)

                            total_incoming = incoming_flow_regular + incoming_flow_fixed

                            # Inventory update constraint for shelters
                            if i in V1:
                                model.addConstr(
                                    I[i, r, t + 1] == I[i, r, t] - outgoing_flow + total_incoming - S[i, t, s] * g_shelter[r],
                                    name="InventoryUpdate_Shelter_%s_%s_%s_%s" % (i, r, t, s)
                                )
                            # Inventory update constraint for pods
                            elif i in V3:
                                model.addConstr(
                                    I[i, r, t + 1] == I[i, r, t] - outgoing_flow + total_incoming - S[i, t, s] * g_pod[r],
                                    name="InventoryUpdate_Pod_%s_%s_%s_%s" % (i, r, t, s)
                                )
                            # Inventory update constraint for warehouses and RSAs
                            else:
                                model.addConstr(
                                    I[i, r, t + 1] == I[i, r, t] - outgoing_flow + total_incoming,
                                    name="InventoryUpdate_%s_%s_%s_%s" % (i, r, t, s))


        # Update demand shortage for all locations, time periods (except t=0), and scenarios
        for i in range(V):
            for t in range(t_bar + 1, T):
                for s in sce:
                    model.addConstr(U[i, t, s] == U[i, t-1, s] - S[i, t, s] + self.demand_processor.d_its[(i, t, s)], 
                                name="DemandShortageUpdate_%s_%s_%s" % (i, t, s))
                    
        # Flow capacity constraint for all links, time periods, and scenarios
        for i in range(V):
            for j in range(V):
                if (i, j) in A:  # Only consider valid links in A
                    for t in range(t_bar, T):
                        for s in sce:
                            # Check if the capacity for this (i, j, t) exists in K_ijt
                            #if i in K_ijt and j in K_ijt[i] and t in K_ijt[i][j]:
                                model.addConstr(
                                    gp.quicksum(w_r[r] * f[i, j, r, t, s] for r in range(R)) <= K_ijt[i][j][t],
                                    name="FlowCapacity_%s_%s_%s_%s" % (i, j, t, s))
                                
        objective = gp.quicksum(
            P[s] * (
                gp.quicksum(
                    gp.quicksum(p_i[i] * U[i, t, s] for i in range(V)) for t in range(t_bar, T)
                ) +
                gp.quicksum(
                    gp.quicksum(
                        gp.quicksum(cost_ir[i][r] * f[i, j, r, t, s] for i in V0 for j in range(V) if (i, j) in A) +
                        gp.quicksum(m_ij[i][j] * f[i, j, r, t, s] for i in range(V) for j in range(V) if (i, j) in A)
                        for r in range(R)
                    ) for t in range(t_bar, T)
                )
            ) for s in sce)
            
        model.setObjective(objective, GRB.MINIMIZE)
        model.setParam("OutputFlag", 0)

        return model, I, U, S, f

    def RH_model_solver(self, model, variables):
        # Solve the model
        model.optimize()

        # Check solution status
        if model.status != GRB.OPTIMAL:
            print(f"Warning: Optimization ended with status {model.status}")
            return None, None

        # Extract objective value
        obj_val = model.objVal

        # Extract values of variables into a dictionary
        variable_values = {
            "I": {(i, r, t): var.X for (i, r, t), var in variables["I"].items()},
            "U": {(i, t, s): var.X for (i, t, s), var in variables["U"].items()},
            "S": {(i, t, s): var.X for (i, t, s), var in variables["S"].items()},
            "f": {(i, j, r, t, s): var.X for (i, j, r, t, s), var in variables["f"].items()},
        }

        # Return everything
        return obj_val, variable_values

    def RH_sol_creator(self, variable_values, t_bar, network_processor, sample_paths, fixed_flow_all):
        
        t_ijt = network_processor.t_ijt

        sol = {}

        # Realized scenario at current time
        s_t = sample_paths[t_bar]
        T = network_processor.T

        if t_bar < T - 1:
            next_t = t_bar + 1
        else:
            next_t = t_bar

        # Inventory at t_bar
        sol["I"] = {
            (i, r, next_t): val
            for (i, r, t), val in variable_values["I"].items()
            if t == next_t
        }

        # Unmet demand at t_bar for the realized scenario only
        sol["U"] = {
            (i, next_t): val
            for (i, t, s), val in variable_values["U"].items()
            if t == next_t and s == s_t
        }
        # ***
        # Fixed flows: from previous rolls that arrive at or after t_bar
        # sol["fixed_flow"] = {}
        # for (i, j, r, t, s), val in variable_values["f"].items():
        #     if s == s_t and t + t_ijt[i][j] == next_t:
        #         sol["fixed_flow"][(i, j, r, t, s)] = val ## !!! t_bar index may cause a problem. Double check!!

        for (i, j, r, t, s), val in variable_values["f"].items():
            if s == s_t and t + t_ijt[i][j] == next_t:  # Arrival is in next time step
                fixed_flow_all[(i, j, r, t, s)] = val


        return sol, fixed_flow_all
        # return sol

    def RH_terminal_model(self, t_terminal, t_bar, sol, s_t, network_processor, demand_processor, sample_paths, fixed_flow_all):
        
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
        P = self.demand_processor.P
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij
        sce = range(self.num_sce)

        # Realized scenario at terminal stage
        s_t = sample_paths[t_terminal]

        # Create the model
        model = gp.Model("Terminal_Model")

        I = {}
        
        for i in range(V):
            for r in range(R):
                I[i, r, t_terminal] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{r}_{t_terminal}")
        
        U = {}
        
        for i in range(V):
            U[i, t_terminal, s_t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_{i}_{t_terminal}_{s_t}")
        
        S = {}
        
        for i in range(V):
            S[i, t_terminal, s_t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{i}_{t_terminal}_{s_t}")
        
        f = {} 
        
        for i in range(V):
            for j in range(V):
                if (i, j) in A:
                    for r in range(R):
                        f[i, j, r, t_terminal, s_t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}_{r}_{t_terminal}_{s_t}")

        # Constraints
        # Initial conditions from sol
        for (i, r, t), val in sol["I"].items():
            model.addConstr(I[i, r, t_terminal] == val, name=f"InitialInventory_{i}_{r}_{t_terminal}")

        for (i, t), val in sol["U"].items():
            model.addConstr(U[i, t_terminal, s_t] == val, name=f"InitialUnmetDemand_{i}_{t_terminal}_{s_t}")

        for i in range(V):
            for r in range(R):
                outgoing = gp.quicksum(f[i, j, r, t_terminal, s_t] for j in range(V) if (i, j) in A)

                # Regular incoming flow (from within RH window)
                # incoming_flow_regular = gp.quicksum(
                #     f[j, i, r, t_terminal - t_ijt[j][i] + 1, s_t]
                #     for j in range(V)
                #     if (j, i) in A and t_terminal - t_ijt[j][i] + 1 >= t_bar)

                # Flows arriving at this roll (from fixed_flow)
                incoming = gp.quicksum(
                    fixed_flow_all.get((j, i, r, t_prime, s_t), 0)
                    for j in range(V)
                    for t_prime in range(t_bar)  # All earlier time steps
                    if (j, i) in A and t_prime + t_ijt[j][i] == t_terminal)
                
                # total_incoming = incoming_flow_regular + incoming_flow_fixed

                if i in V1:
                    model.addConstr(
                        I[i, r, t_terminal] - outgoing + incoming - S[i, t_terminal, s_t] * g_shelter[r] >= 0,
                        name=f"SafeInventory_Shelter_{i}_{r}")  
                elif i in V3:
                    model.addConstr(
                        I[i, r, t_terminal] - outgoing + incoming - S[i, t_terminal, s_t] * g_pod[r] >= 0,
                        name=f"SafeInventory_PoD_{i}_{r}")
                else:
                    model.addConstr(
                        I[i, r, t_terminal] - outgoing + incoming >= 0,
                        name=f"SafeInventory_{i}_{r}")

        # for i in range(V):
        #     model.addConstr(
        #         U[i] == sol["U"][(i, t)]- S[i] + self.demand_processor.d_its[(i, t, s_t)],
        #         name=f"DemandUpdate_{i}")

        for i in range(V):
            for j in range(V):
                if (i, j) in A:
                    model.addConstr(
                        gp.quicksum(w_r[r] * f[i, j, r, t_terminal, s_t] for r in range(R)) <= K_ijt[i][j][t_terminal],
                        name=f"FlowCap_{i}_{j}_{t_terminal}")
                    
        # Objective function
        objective = (
            gp.quicksum(p_i[i] * U[i, t_terminal, s_t] for i in range(V)) +
            gp.quicksum(
                cost_ir[i][r] * f[i, j, r, t_terminal, s_t]
                for i in V0 for j in range(V) if (i, j) in A
                for r in range(R)
            ) +
            gp.quicksum(
                m_ij[i][j] * f[i, j, r, t_terminal, s_t]
                for i in range(V) for j in range(V) if (i, j) in A
                for r in range(R)
            )
        )

        model.setObjective(objective, GRB.MINIMIZE)
        model.setParam("OutputFlag", 0)
        model.update()

        return model, I, U, S, f
    
    def RH_evaluation(self, sample_paths, demand_processor, network_processor):

        T = network_processor.T
        P = self.demand_processor.P
        V = self.network_processor.V
        V0 = self.network_processor.V0
        R = self.network_processor.R
        A = self.network_processor.A
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij
        nbOS = demand_processor.nbOS

        start_time = time.time()
        # Load sample paths from CSV
        OS_paths = pd.read_csv(sample_paths).values
        T = OS_paths.shape[1]
        total_costs = np.zeros(nbOS)  # Total cost for each sample path

        for k in range(nbOS):
            print("Sample path=", k)
            sample_paths = OS_paths[k]  # 1D array for path k
            fixed_flow_all = {}
            t_bar = 0
            sol = {}  # Initial state
            cost_k = 0  # Total cost for sample path k

            # Rolling loop over time steps
            while t_bar < T:
                print("t_bar=", t_bar)
                s_t = sample_paths[t_bar]  # Realized scenario at current time

                # Build the model for current roll
                model, I, U, S, f = self.RH_model_creator(t_bar, s_t, network_processor, demand_processor, sol, fixed_flow_all)
                # model, I, U, S, f = self.RH_model_creator(t_bar, s_t, network_processor, demand_processor, sol)
                # Solve the model
                obj_val, variable_values = self.RH_model_solver(model, {"I": I, "U": U, "S": S, "f": f})

                if obj_val is None:
                    print(f"Skipping sample path {k} at t_bar={t_bar} due to infeasibility.")
                    exit(0)
                
                # Calculate cost only for current t_bar
                penalty_cost = sum(p_i[i] * variable_values["U"][i, t_bar, s_t] for i in range(V))

                transport_cost = sum(
                    cost_ir[i][r] * variable_values["f"].get((i, j, r, t_bar, s_t), 0)
                    for i in V0 for j in range(V) for r in range(R) if (i, j) in A
                ) + sum(
                    m_ij[i][j] * variable_values["f"].get((i, j, r, t_bar, s_t), 0)
                    for i in range(V) for j in range(V) for r in range(R) if (i, j) in A)

                cost_k += penalty_cost + transport_cost

                # Accumulate fixed flows made at t_bar
                for (i, j, r, t, s), val in variable_values["f"].items():
                    if t == t_bar:
                        fixed_flow_all[(i, j, r, t, s)] = val

                # Update the rolling state for next time step (should be before model creation, because we need updated sol to create the model every time)
                sol, fixed_flow_all = self.RH_sol_creator(variable_values, t_bar, network_processor, sample_paths, fixed_flow_all)
                # sol = self.RH_sol_creator(variable_values, t_bar, network_processor, sample_paths)
                
                # If at the last roll, apply terminal stage logic (placeholder)
                if t_bar == T - 1:

                    #Solve terminal stage model
                    terminal_model, I_term, U_term, S_term, f_term = self.RH_terminal_model(T - 1, t_bar, sol, s_t, network_processor, demand_processor, sample_paths, fixed_flow_all)

                    obj_term, _ = self.RH_model_solver(terminal_model, {"I": I_term, "U": U_term, "S": S_term, "f": f_term})

                    if obj_term is not None:
                        cost_k += obj_term
                    else:
                        print(f"Terminal stage for sample path {k} was infeasible.")
                        exit(0)   

                # Advance to the next time step
                t_bar += 1

            # Save the cost for this sample path
            total_costs[k] = cost_k

        # Average cost across all sample paths
        average_cost = np.mean(total_costs)

        return average_cost, total_costs

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

        P = self.demand_processor.P
        V = self.network_processor.V
        R = self.network_processor.R
        T = self.network_processor.T
        d_its = self.demand_processor.d_its
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
