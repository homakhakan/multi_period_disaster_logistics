import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from demand_data import DemandData
import time
import os

### Extended
class TwoStageSPModel:
    def __init__(self, network_processor, demand_processor, num_sce):
        self.network_processor = network_processor  # Store network data
        self.demand_processor = demand_processor  # Store demand data
        # if not hasattr(self.demand_processor, 'd_its_original'):
        #     self.demand_processor.d_its_original = self.demand_processor.d_its
        # if not hasattr(self.demand_processor, 'P_original'):
        #     self.demand_processor.P_original = self.demand_processor.P.copy()

        self.num_sce = num_sce  # Use full scenarios for other models

        # Initialize fixed_flow_all for use across RH-related functions
        self.fixed_flow_all = {}

    def ext_model(self, is_mvp=False):

        if is_mvp:
            self.demand_processor.d_its = self.demand_processor.d_its_expected
            self.demand_processor.P = [1]
            self.num_sce = 1
        else:
            # self.demand_processor.d_its = self.demand_processor.d_its_original
            # self.demand_processor.P = self.demand_processor.P_original
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
        m.setParam("OutputFlag", 0)
        m.update()
        return m, I

    def OOS_2SSP(self, fixed_I_irt):
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

        # Build the model (variables are free initially)
        model, I_vars = self.ext_model(is_mvp=False)

        # Fix each I[i, r, t] to its corresponding value using equality constraints
        for (i, r, t), val in fixed_I_irt.items():
            model.addConstr(I_vars[i, r, t] == val, name=f"fix_I_{i}_{r}_{t}")

        # Solve the model
        model.optimize()

        # Return objective value if solved successfully
        if model.status == GRB.OPTIMAL:
            return model.objVal
        else:
            print(f"[Warning] Model not optimal. Status: {model.status}")
            return float("inf")

    def extract_all_inventory(self, I_vars):
        inventory_values = {}
        for i in range(self.network_processor.V):
            for r in range(self.network_processor.R):
                for t in range(self.network_processor.T):
                    inventory_values[(i, r, t)] = I_vars[i, r, t].X
        return inventory_values

    def RH_model_creator(self, t_bar, sol):

        # Builds the Gurobi model for the current rolling horizon window starting at time t_bar.
        # Returns the model and the key variable dictionaries (I, U, S, f).

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
                                f[j, i, r, t - (t_ijt[j][i] + 1), s]
                                for j in range(V)
                                if (j, i) in A and t - (t_ijt[j][i] + 1) >= t_bar)
                            
                            incoming_flow_fixed = gp.quicksum(
                            self.fixed_flow_all.get((j, i, r, t_prime, s), 0)
                            for j in range(V)
                            for t_prime in range(t_bar)  # Flows made before t_bar
                            if (j, i) in A and t_prime + (t_ijt[j][i] + 1) == t)

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
                    model.addConstr(
                        U[i, t, s] == U[i, t - 1, s] - S[i, t, s] + self.demand_processor.d_its[(i, t, s)],
                        name=f"DemandShortageUpdate_{i}_{t}_{s}")

                    
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
                                
        # Add nonanticipativity constraints: all scenarios must match scenario 0 at t_bar
        for i in range(V):
            for j in range(V):
                if (i, j) in A:
                    for r in range(R):
                        for s in range(1, self.num_sce):  # skip s=0 (used as reference)
                            model.addConstr(
                                f[i, j, r, t_bar, s] == f[i, j, r, t_bar, 0],
                                name=f"NonAntic_f_{i}_{j}_{r}_{t_bar}_{s}")

                                
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
            return None, None

        # Extract objective value
        obj_val = model.objVal

        # Extract values of variables into a dictionary
        # Determine if this is terminal model (U keys are 2-tuples)
        is_terminal = all(len(k) == 2 for k in variables["U"].keys())

        if is_terminal:
            variable_values = {
                "I": {(i, r, t): var.X for (i, r, t), var in variables["I"].items()},
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

    def RH_sol_creator(self, variable_values, t_bar, s_realized):
        """
        Extracts inventory and unmet demand values at time t_bar + 1 from the solved model for the realized scenario. 
        Updates the solution dictionary (sol) with these values and returns it
        Used for initializing the next rolling horizon stage.
        """
        sol = {}

        next_t = t_bar + 1

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
            if t == next_t and s == s_realized
        }

        return sol

    def RH_terminal_model(self, sol, s_t, sample_paths):
        """
        Builds and returns the deterministic terminal-stage model for the rolling horizon framework,
        covering the period from t_L + 1 to T after hurricane landfall.
        """

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
        t_L = self.network_processor.t_L

        # Create the model
        model = gp.Model("Terminal_Model")

        I = {}
        
        for i in range(V):
            for r in range(R):
                for t in range(t_L + 1, T):
                    I[i, r, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{r}_{t}")
        
        U = {}
        
        for i in range(V):
            for t in range(t_L + 1, T):
                U[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_{i}_{t}")
        
        S = {}
        
        for i in range(V):
            for t in range(t_L + 1, T):
                S[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{i}_{t}")
        
        f = {} 
        
        for i in range(V):
            for j in range(V):
                if (i, j) in A:
                    for r in range(R):
                        for t in range(t_L + 1, T):
                            f[i, j, r, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}_{r}_{t}")

        # Constraints
        # Initial conditions from sol
        for i in range(V):
            for r in range(R):
                model.addConstr(
                    I[i, r, t_L + 1] == sol["I"][i, r, t_L + 1],
                    name=f"InitInv_Terminal_{i}_{r}_{t_L + 1}")

        for i in range(V):
            model.addConstr(
                U[i, t_L + 1] == sol["U"][i, t_L + 1],
                name=f"InitUnmet_Terminal_{i}_{t_L + 1}")
            
        # Flow balance
        for i in range(V):
                for r in range(R):
                    for t in range(t_L + 1, T - 1):
                        outgoing_flow = gp.quicksum(f[i, j, r, t] for j in range(V) if (i, j) in A)
                        
                        # # Regular incoming flow (from inside RH window)
                        # incoming_flow_regular = gp.quicksum(
                        #     f[j, i, r, t - (t_ijt[j][i] + 1)]
                        #     for j in range(V)
                        #     if (j, i) in A and t - (t_ijt[j][i] + 1) >= t_L)

                        incoming_flow_regular = gp.quicksum(
                            f[j, i, r, t_prime]
                            for j in range(V)
                            if (j, i) in A
                            for t_prime in range(t_L + 1, t)  # only valid terminal time range
                            if t_prime + (t_ijt[j][i] + 1) == t)
                        
                        incoming_flow_fixed = gp.quicksum(
                        self.fixed_flow_all.get((j, i, r, t_prime), 0)
                        for j in range(V)
                        for t_prime in range(t_L)  # Flows made before t_L
                        if (j, i) in A and t_prime + (t_ijt[j][i] + 1) == t)

                        total_incoming = incoming_flow_regular + incoming_flow_fixed

                        # Inventory update constraint for shelters
                        if i in V1:
                            model.addConstr(
                                I[i, r, t + 1] == I[i, r, t] - outgoing_flow + total_incoming - S[i, t] * g_shelter[r],
                                name="InventoryUpdate_Shelter_%s_%s_%s" % (i, r, t)
                            )
                        # Inventory update constraint for pods
                        elif i in V3:
                            model.addConstr(
                                I[i, r, t + 1] == I[i, r, t] - outgoing_flow + total_incoming - S[i, t] * g_pod[r],
                                name="InventoryUpdate_Pod_%s_%s_%s" % (i, r, t)
                            )
                        # Inventory update constraint for warehouses and RSAs
                        else:
                            model.addConstr(
                                I[i, r, t + 1] == I[i, r, t] - outgoing_flow + total_incoming,
                                name="InventoryUpdate_%s_%s_%s" % (i, r, t))

        for i in range(V):
            for t in range(t_L + 2, T):  # T is exclusive in range, so this covers t = t_L+2 to T-1
                model.addConstr(
                    U[i, t] == U[i, t - 1] - S[i, t] + self.demand_processor.d_its[(i, t)], # realized demand using s_t
                    name=f"DemandShortageUpdate_Terminal_{i}_{t}")

        for i in range(V):
            for j in range(V):
                if (i, j) in A:
                    for t in range(t_L + 1, T):  # T is exclusive, so this gives t_L+1 to T-1
                        model.addConstr(
                            gp.quicksum(w_r[r] * f[i, j, r, t] for r in range(R)) <= K_ijt[i][j][t],
                            name=f"FlowCapacity_Terminal_{i}_{j}_{t}")

                    
        # Objective function
        objective = (
            gp.quicksum(p_i[i] * U[i, t] for i in range(V) for t in range(t_L + 1, T)) +
            gp.quicksum(
                cost_ir[i][r] * f[i, j, r, t]
                for i in V0 for j in range(V) if (i, j) in A
                for r in range(R) for t in range(t_L + 1, T)
            ) +
            gp.quicksum(
                m_ij[i][j] * f[i, j, r, t]
                for i in range(V) for j in range(V) if (i, j) in A
                for r in range(R) for t in range(t_L + 1, T)))


        model.setObjective(objective, GRB.MINIMIZE)
        model.setParam("OutputFlag", 0)
        model.update()

        return model, I, U, S, f
    
    def calculate_cost_rh(self, variable_values, t_bar, s_t):
        """
        Calculates the total cost (penalty + transportation) for the deterministic terminal model
        Returns Total cost at time t_bar.
        """
        V = self.network_processor.V
        V0 = self.network_processor.V0
        R = self.network_processor.R
        A = self.network_processor.A
        p_i = self.network_processor.p_i
        cost_ir = self.network_processor.cost_ir
        m_ij = self.network_processor.m_ij

        # Penalty cost for unmet demand
        penalty_cost = sum(
            p_i[i] * variable_values["U"].get((i, t_bar, s_t), 0) for i in range(V))

        # Transportation cost from source nodes
        transport_cost = sum(
            cost_ir[i][r] * variable_values["f"].get((i, j, r, t_bar, s_t), 0)
            for i in V0 for j in range(V) for r in range(R) if (i, j) in A)

        # Additional per-mile transportation cost
        transport_cost += sum(
            m_ij[i][j] * variable_values["f"].get((i, j, r, t_bar, s_t), 0)
            for i in range(V) for j in range(V) for r in range(R) if (i, j) in A)

        return penalty_cost + transport_cost

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
        from demand_generation import data, in_sample

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

        for k in range(nbOS):
            print("Sample path=", k)
            sample_path_row = OS_paths[k]  # 1D array for path k
            self.fixed_flow_all = {}
            t_bar = 0
            sol = {}  # Initial state
            cost_k = 0  # Total cost for sample path k

            # Rolling loop over time steps
            while t_bar <= t_L:
                print("t_bar=", t_bar)
                s_t = eval(sample_path_row[t_bar])  # Converts '(-23.55, 0)' to (-23.55, 0)

                # Generate the demand at t_bar only based on the realized current state of the hurricane
                demand_processor.rh_generate_current_demand(t_bar, s_t, network_processor, data)

                # Generate multiple in-sample scenario paths starting from the current time step t_bar and realized hurricane state s_t.
                # Each scenario is a full trajectory of future hurricane states (track error, intensity) used to simulate uncertainty from t_bar onward.
                in_sample_paths = in_sample(t_bar, s_t, self.num_sce)

                # Identify which in-sample scenario matches the realized state s_t
                s_realized = None
                for s in range(self.num_sce):
                    if tuple(in_sample_paths[s, t_bar]) == s_t:
                        s_realized = s
                        break

                if s_realized is None:
                    raise ValueError(f"No in-sample scenario matches realized state s_t = {s_t} at t_bar = {t_bar}")

                # Generate the demand at t_bar + 1 to t_L based on the realized state of the hurricane at t_bar
                demand_processor.rh_generate_future_demand(t_bar, s_t, t_L, network_processor, in_sample_paths)  

                demand_processor.rh_generate_post_landfall_demand(t_L, in_sample_paths, network_processor)
                    
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
                        self.fixed_flow_all[(i, j, r, t, s)] = val
                
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
