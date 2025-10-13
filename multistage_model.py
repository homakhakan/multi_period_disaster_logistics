import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import msp_helper
import time
from openpyxl import load_workbook
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

class MultiStageSPModel:
    def __init__(self, network_processor, demand_processor, params, travel_time, last_stage):
        self.network_processor = network_processor
        self.demand_processor = demand_processor
        self.params = params
        self.travel_time = travel_time
        self.last_stage = last_stage
        self.smallest_trans_prob = float(params["smallest_trans_prob"])
        self.cutviol = float(params["CUTVIOL"])
        self.max_iter = int(params["MAX_ITER"])
        self.stall = int(params["STALL"])
        self.cutviol_maxiter = int(params["CUTVIOL_MAXITER"])
        self.time_limit = float(params["TIME_LIMIT"])
        self.lower_bound = float(params["LOWER_BOUND"])
        self.rel_gap = float(params["relative_gap"])

        self.V = network_processor.V
        self.V0 = network_processor.V0
        self.V1 = network_processor.V1
        self.V2 = network_processor.V2
        self.V3 = network_processor.V3
        self.R = network_processor.R
        self.A = network_processor.A
        self.g_shelter = network_processor.g_shelter
        self.g_pod = network_processor.g_pod
        self.K_ijt = network_processor.K_ijt
        self.w_r = network_processor.w_r
        self.p_i = network_processor.p_i
        self.cost_ir = network_processor.cost_ir
        self.m_ij = network_processor.m_ij
        self.T = network_processor.T
        self.T1 = network_processor.T1
        self.t_ijt = network_processor.t_ijt
        self.t_L = network_processor.t_L 
        self.tt_max = network_processor.tt_max

        # Define delta indicator
        if self.travel_time == 2:
            self.delta = []        
            for (i, j) in self.A:
                for t in range(self.T): 
                    t_prime = t - self.t_ijt[i][j] - 1
                    if 0 <= t_prime:          
                        self.delta.append((i, j, t_prime, t))

            # Convert to set for fast membership checking
            self.delta = set(self.delta)

            # print(len(self.delta))
            # for delta in self.delta:
            #     print(delta)

        # Markov chain data
        transition_path = "Data/pi_mssp_d_ST_track_15.json"
        self.transition_dict = msp_helper.load_transition_dict(transition_path)
        self.nodeLists = msp_helper.build_node_lists(self.transition_dict)

    def stage_t_state_k_problem(self, t, k):
        """
        Build the optimization model for *one* node (t, k) in the scenario tree.
        t : Stage index (time period).
        k : Markov-chain state realised
        """
        
        # Create a model for each stage t
        m = gp.Model(f"Stage_{t}_state_{k}_model")
        m.setParam("OutputFlag", 0)
        m.modelSense = GRB.MINIMIZE

        # Define state variables
        I = {}
        for i in range(self.V):
            for r in range(self.R):
                I[i, r] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{r}")

        U = {}
        for i in range(self.V):
            U[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_{i}")

        if self.travel_time == 2:
            f1 = {}
            for i in range(self.V):
                for j in range(self.V):
                    if (i, j) in self.A:
                        for r in range(self.R):
                            for t_prime in range(t+1):
                                f1[i, j, r, t_prime] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f1_{i}_{j}_{r}_{t_prime}")

        # Define local variables
        S = {}
        for i in range(self.V):
            S[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{i}")

        f = {}      
        for i in range(self.V):
            for j in range(self.V):
                if (i, j) in self.A:
                    for r in range(self.R):
                        f[i, j, r] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}_{r}")

        # Always create theta, but fix it later to 0 in the last stage
        theta = m.addVar(lb=0, name="theta")

        # Constraints
        # Demand shortage update
        demand_shortage_constr1 = {}
        for i in range(self.V):
            demand_shortage_constr1[i] = m.addConstr(U[i] + S[i] <= 0, name=f"DemandShortageUpdate1_{i}")

        demand_shortage_constr2 = {}
        for i in range(self.V):
            demand_shortage_constr2[i] = m.addConstr(U[i] + S[i] >= 0, name=f"DemandShortageUpdate2_{i}")

        # Inventory update
        if self.travel_time == 1:
            inventory_update_constr1 = {}
            for i in range(self.V):
                for r in range(self.R):
                    # Determine g_r coefficient
                    if i in self.V1:            
                        g_coeff = self.g_shelter[r] # shelter
                    elif i in self.V3:          
                        g_coeff = self.g_pod[r] # PoD
                    else:                  
                        g_coeff = 0 # warehouse or RSA

                    # Adjust RHS based on t
                    if t == 0:
                        inventory_update_constr1[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) - gp.quicksum(f[j, i, r] for j in range(self.V) if (j, i) in self.A)
                                                                    + S[i] * g_coeff <= self.network_processor.I_ir0[i][r], name=f"InventoryUpdate1_{i}_{r}_t0")
                    else:
                        inventory_update_constr1[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) - gp.quicksum(f[j, i, r] for j in range(self.V) if (j, i) in self.A)
                                                                    + S[i] * g_coeff <= 0, name=f"InventoryUpdate1_{i}_{r}_t{t}")   

            inventory_update_constr2 = {}
            for i in range(self.V):
                for r in range(self.R):
                    # Determine g_r coefficient
                    if i in self.V1:            
                        g_coeff = self.g_shelter[r] # shelter
                    elif i in self.V3:          
                        g_coeff = self.g_pod[r] # PoD
                    else:                  
                        g_coeff = 0 # warehouse or RSA

                    # Adjust RHS based on t
                    if t == 0:
                        inventory_update_constr2[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) - gp.quicksum(f[j, i, r] for j in range(self.V) if (j, i) in self.A)
                                                                    + S[i] * g_coeff >= self.network_processor.I_ir0[i][r], name=f"InventoryUpdate2_{i}_{r}_t0")
                    else:
                        inventory_update_constr2[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) - gp.quicksum(f[j, i, r] for j in range(self.V) if (j, i) in self.A)
                                                                    + S[i] * g_coeff >= 0, name=f"InventoryUpdate2_{i}_{r}_t{t}")      
        
        elif self.travel_time == 2:
            inventory_update_constr1 = {}
            for i in range(self.V):
                for r in range(self.R):
                    # Determine g_r coefficient
                    if i in self.V1:            
                        g_coeff = self.g_shelter[r] # shelter
                    elif i in self.V3:          
                        g_coeff = self.g_pod[r] # PoD
                    else:                  
                        g_coeff = 0 # warehouse or RSA

                    # Adjust RHS based on t
                    if t == 0:
                        inventory_update_constr1[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) 
                                                                    + S[i] * g_coeff <= self.network_processor.I_ir0[i][r], name=f"InventoryUpdate1_{i}_{r}_t0")
                    else:
                        inventory_update_constr1[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) 
                                                                    + S[i] * g_coeff <= 0, name=f"InventoryUpdate1_{i}_{r}_t{t}") 
                        
            inventory_update_constr2 = {}
            for i in range(self.V):
                for r in range(self.R):
                    # Determine g_r coefficient
                    if i in self.V1:            
                        g_coeff = self.g_shelter[r] # shelter
                    elif i in self.V3:          
                        g_coeff = self.g_pod[r] # PoD
                    else:                  
                        g_coeff = 0 # warehouse or RSA

                    # Adjust RHS based on t
                    if t == 0:
                        inventory_update_constr2[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) 
                                                                    + S[i] * g_coeff >= self.network_processor.I_ir0[i][r], name=f"InventoryUpdate2_{i}_{r}_t0")
                    else:
                        inventory_update_constr2[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) 
                                                                    + S[i] * g_coeff >= 0, name=f"InventoryUpdate2_{i}_{r}_t{t}") 

            # In-transit flow update to the next stage 
            in_transit_update_constr1 = {}
            for (i, j) in self.A:
                for r in range(self.R):
                    # t′ < t: Carry forward past flows (to be updated later)
                    for t_prime in range(t):
                        in_transit_update_constr1[i, j, r, t_prime] = m.addConstr(f1[i, j, r, t_prime] <= 0, name=f"CarryTransit1_{i}_{j}_{r}_{t_prime}_to_{t}") 
                

            in_transit_update_constr2 = {}
            for (i, j) in self.A:
                for r in range(self.R):
                    # t′ < t: Carry forward past flows (to be updated later)
                    for t_prime in range(t):
                        in_transit_update_constr2[i, j, r, t_prime] = m.addConstr(f1[i, j, r, t_prime] >= 0, name=f"CarryTransit2_{i}_{j}_{r}_{t_prime}_to_{t}")
                    

            # In-transit flow initialization at current stage t (t' = t)
            init_in_transit_constr = {}
            for (i, j) in self.A:
                for r in range(self.R):
                    init_in_transit_constr[i, j, r] = m.addConstr(f1[i, j, r, t] == f[i, j, r], name=f"InitInTransit_{i}_{j}_{r}_t{t}")

        # Flow capacity
        flow_capacity_constr = {}
        for (i, j) in self.A:
            flow_capacity_constr[i, j] = m.addConstr(gp.quicksum(self.w_r[r] * f[i, j, r] for r in range(self.R)) <= self.K_ijt[i][j][t], name=f"FlowCapacity_{i}_{j}_{t}")

        # Objective
        objective = (
            gp.quicksum(self.p_i[i] * U[i] for i in range(self.V)) 
            + gp.quicksum(
                gp.quicksum(self.cost_ir[i][r] * f[i, j, r] for i in self.V0 for j in range(self.V) if (i, j) in self.A)
                + gp.quicksum(self.m_ij[i][j] * f[i, j, r] for (i, j) in self.A) for r in range(self.R))
            + theta
        )
        

        m.setObjective(objective, GRB.MINIMIZE)
        m.update()

        if self.travel_time == 1:
            return m, I, U, f, S, theta, demand_shortage_constr1, demand_shortage_constr2, inventory_update_constr1, inventory_update_constr2, flow_capacity_constr
            
        elif self.travel_time == 2:
            return m, I, U, f, S, f1, theta, demand_shortage_constr1, demand_shortage_constr2, inventory_update_constr1, inventory_update_constr2, flow_capacity_constr, in_transit_update_constr1, in_transit_update_constr2, init_in_transit_constr
            
    
    def build_models(self):
        """ Define optimization models for each (t, k)"""

        # Prepare dictionaries
        if self.travel_time == 1:
            self.m = {}
            self.I = {}
            self.U = {}        
            self.f = {}
            self.S = {}
            self.theta = {}
            self.demand_shortage_constr1 = {}
            self.demand_shortage_constr2 = {}
            self.inventory_update_constr1 = {}
            self.inventory_update_constr2 = {}
            self.flow_capacity_constr = {}

        elif self.travel_time == 2:
            self.m = {}
            self.I = {}
            self.U = {}        
            self.f = {}
            self.S = {}
            self.f1 = {}
            self.theta = {}
            self.demand_shortage_constr1 = {}
            self.demand_shortage_constr2 = {}
            self.inventory_update_constr1 = {}
            self.inventory_update_constr2 = {}
            self.flow_capacity_constr = {}
            self.in_transit_update_constr1 = {}
            self.in_transit_update_constr2 = {}
            self.init_in_transit_constr = {}

        # Loop over pre-landfall stochastic stages (t = 0 … 9)
        if self.last_stage == 1:
            for t in range(self.T1):
                for k in self.nodeLists[t]: # Every reachable Markov state at time t
                    if self.travel_time == 1:
                        (
                            self.m[t, k],
                            self.I[t, k],
                            self.U[t, k],   
                            self.f[t, k],
                            self.S[t, k],
                            self.theta[t, k],
                            self.demand_shortage_constr1[t, k],
                            self.demand_shortage_constr2[t, k],
                            self.inventory_update_constr1[t, k],
                            self.inventory_update_constr2[t, k],
                            self.flow_capacity_constr[t, k]
                        ) = self.stage_t_state_k_problem(t, k)
                    elif self.travel_time == 2:
                        (
                            self.m[t, k],
                            self.I[t, k],
                            self.U[t, k],
                            self.f[t, k],
                            self.S[t, k],
                            self.f1[t, k],
                            self.theta[t, k],
                            self.demand_shortage_constr1[t, k],
                            self.demand_shortage_constr2[t, k],
                            self.inventory_update_constr1[t, k],
                            self.inventory_update_constr2[t, k],
                            self.flow_capacity_constr[t, k],
                            self.in_transit_update_constr1[t, k],
                            self.in_transit_update_constr2[t, k],
                            self.init_in_transit_constr[t, k],
                        ) = self.stage_t_state_k_problem(t, k)
                if t == self.t_L:
                    self.theta[t, k].setAttr(GRB.Attr.UB, 0) # Fix theta = 0 at last stage

        elif self.last_stage == 2:
            for t in range(self.T1 - 1):
                for k in self.nodeLists[t]: # Every reachable Markov state at time t
                    if self.travel_time == 1:
                        (
                            self.m[t, k],
                            self.I[t, k],
                            self.U[t, k],   
                            self.f[t, k],
                            self.S[t, k],
                            self.theta[t, k],
                            self.demand_shortage_constr1[t, k],
                            self.demand_shortage_constr2[t, k],
                            self.inventory_update_constr1[t, k],
                            self.inventory_update_constr2[t, k],
                            self.flow_capacity_constr[t, k]
                        ) = self.stage_t_state_k_problem(t, k)
                    elif self.travel_time == 2:
                        (
                            self.m[t, k],
                            self.I[t, k],
                            self.U[t, k],
                            self.f[t, k],
                            self.S[t, k],
                            self.f1[t, k],
                            self.theta[t, k],
                            self.demand_shortage_constr1[t, k],
                            self.demand_shortage_constr2[t, k],
                            self.inventory_update_constr1[t, k],
                            self.inventory_update_constr2[t, k],
                            self.flow_capacity_constr[t, k],
                            self.in_transit_update_constr1[t, k],
                            self.in_transit_update_constr2[t, k],
                            self.init_in_transit_constr[t, k],
                        ) = self.stage_t_state_k_problem(t, k)
    
    def last_stage_model(self, k_t):

        m = gp.Model("last_stage")
        m.setParam("OutputFlag", 0)
        m.modelSense = GRB.MINIMIZE

        # Define variables
        I = {}
        for i in range(self.V):
            for r in range(self.R):
                for t in range(self.t_L, self.T): # t = 9, 10, ..., 15
                    I[i, r, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{r}_{t}")

        U = {}
        for i in range(self.V):
            for t in range(self.t_L, self.T):
                U[i, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_{i}_{t}")

        S = {}
        for i in range(self.V):
            for t in range(self.t_L, self.T):
                S[i, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{i}_{t}")

        
        f = {}      
        for i in range(self.V):
            for j in range(self.V):
                if (i, j) in self.A:
                    for r in range(self.R):
                        for t in range(self.t_L, self.T):
                            f[i, j, r, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}_{r}_{t}")

        # if self.travel_time == 2: ***
        #     f1 = {}
        #     for i in range(self.V):
        #         for j in range(self.V):
        #             if (i, j) in self.A:
        #                 for r in range(self.R):
        #                     for t in range(self.t_L, self.T):        # t = 9, ..., 15
        #                         for t_prime in range(t+1):             # t' <= t
        #                             f1[i, j, r, t_prime, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f1_{i}_{j}_{r}_{t_prime}_{t}")
        
        # Constraints
        # Demand shortage init
        demand_shortage_constr1 = {}
        for i in range(self.V):
            demand_shortage_constr1[i] = m.addConstr(U[i, self.t_L] + S[i, self.t_L] <= 0, name=f"DemandShortageUpdate1_{i}")

        demand_shortage_constr2 = {}
        for i in range(self.V):
            demand_shortage_constr2[i] = m.addConstr(U[i, self.t_L] + S[i, self.t_L] >= 0, name=f"DemandShortageUpdate2_{i}")

        # Demand after landfall, only for pods
        post_landfall_demand = self.demand_processor.msp_post_landfall_demand(k_t, self.network_processor)

        # for (i, t), val in post_landfall_demand.items():
        #     if i not in self.V3 and val > 1e-6:  # small threshold to ignore rounding
        #         print(f"Unexpected demand at non-PoD node {i} at t={t}: {val}")
        
        # # print("Non-zero post-landfall demand values:")
        # for (i, t), val in post_landfall_demand.items():
        #     if val > 1e-3:
        #         print(f"demand[{i},{t}] = {val:.2f}")
        
        # Demand shortage post landfall
        demand_shortage_post = {}
        for i in range(self.V):
            for t in range(self.t_L + 1, self.T):
                demand_shortage_post[i, t] = m.addConstr(U[i, t] == U[i, t - 1] - S[i, t] + post_landfall_demand[(i, t)], name=f"DemandShortagePost_{i}_{t}")
        
        if self.travel_time == 1:
            # Inventory update init
            inventory_update_constr1 = {}
            for i in range(self.V):
                for r in range(self.R):
                    # Determine g_r coefficient
                    if i in self.V1:            
                        g_coeff = self.g_shelter[r] # shelter
                    elif i in self.V3:          
                        g_coeff = self.g_pod[r] # PoD
                    else:                  
                        g_coeff = 0 # warehouse or RSA
                    inventory_update_constr1[i, r] = m.addConstr(I[i, r, self.t_L] + gp.quicksum(f[i, j, r, self.t_L] for j in range(self.V) if (i, j) in self.A) 
                                                                 - gp.quicksum(f[j, i, r, self.t_L] for j in range(self.V) if (j, i) in self.A)
                                                                 + S[i, self.t_L] * g_coeff <= 0, name=f"InventoryUpdate1_{i}_{r}_{self.t_L}")

            inventory_update_constr2 = {}
            for i in range(self.V):
                for r in range(self.R):
                    # Determine g_r coefficient
                    if i in self.V1:            
                        g_coeff = self.g_shelter[r] # shelter
                    elif i in self.V3:          
                        g_coeff = self.g_pod[r] # PoD
                    else:                  
                        g_coeff = 0 # warehouse or RSA
                    inventory_update_constr2[i, r] = m.addConstr(I[i, r, self.t_L] + gp.quicksum(f[i, j, r, self.t_L] for j in range(self.V) if (i, j) in self.A)
                                                                 - gp.quicksum(f[j, i, r, self.t_L] for j in range(self.V) if (j, i) in self.A)
                                                                 + S[i, self.t_L] * g_coeff >= 0, name=f"InventoryUpdate2_{i}_{r}_{self.t_L}")
            
            # Inventory update post landfall
            inventory_update_post= {}
            for i in range(self.V):
                for r in range(self.R):
                    for t in range(self.t_L + 1, self.T):
                        # Determine g_r coefficient
                        if i in self.V1:            
                            g_coeff = self.g_shelter[r] # shelter
                        elif i in self.V3:          
                            g_coeff = self.g_pod[r] # PoD
                        else:                  
                            g_coeff = 0 # warehouse or RSA
                        inventory_update_post[i, r, t] = m.addConstr(I[i, r, t] == I[i, r, t - 1] - gp.quicksum(f[i, j, r, t] for j in range(self.V) if (i, j) in self.A) 
                                                                    + gp.quicksum(f[j, i, r, t] for j in range(self.V) if (j, i) in self.A)
                                                                    - S[i, t] * g_coeff, name=f"InventoryUpdatePost2_{i}_{r}_{t}")     
        
        elif self.travel_time == 2:
            # Inventory update init
            inventory_update_constr1 = {}
            for i in range(self.V):
                for r in range(self.R):
                    # Determine g_r coefficient
                    if i in self.V1:            
                        g_coeff = self.g_shelter[r] # shelter
                    elif i in self.V3:          
                        g_coeff = self.g_pod[r] # PoD
                    else:                  
                        g_coeff = 0 # warehouse or RSA
                    inventory_update_constr1[i, r] = m.addConstr(I[i, r, self.t_L] + gp.quicksum(f[i, j, r, self.t_L] for j in range(self.V) if (i, j) in self.A) 
                                                                 + S[i, self.t_L] * g_coeff <= 0, name=f"InventoryUpdate1_{i}_{r}_{self.t_L}")

            inventory_update_constr2 = {}
            for i in range(self.V):
                for r in range(self.R):
                    # Determine g_r coefficient
                    if i in self.V1:            
                        g_coeff = self.g_shelter[r] # shelter
                    elif i in self.V3:          
                        g_coeff = self.g_pod[r] # PoD
                    else:                  
                        g_coeff = 0 # warehouse or RSA
                    inventory_update_constr2[i, r] = m.addConstr(I[i, r, self.t_L] + gp.quicksum(f[i, j, r, self.t_L] for j in range(self.V) if (i, j) in self.A) 
                                                                 + S[i, self.t_L] * g_coeff >= 0, name=f"InventoryUpdate2_{i}_{r}_{self.t_L}")

            # # Inventory update post landfall ***
            # inventory_update_post= {}
            # for i in range(self.V):
            #     for r in range(self.R):
            #         for t in range(self.t_L + 1, self.T):
            #             # Determine g_r coefficient
            #             if i in self.V1:            
            #                 g_coeff = self.g_shelter[r] # shelter
            #             elif i in self.V3:          
            #                 g_coeff = self.g_pod[r] # PoD
            #             else:                  
            #                 g_coeff = 0 # warehouse or RSA
            #             inventory_update_post[i, r, t] = m.addConstr(I[i, r, t] == I[i, r, t - 1] - gp.quicksum(f[i, j, r, t] for j in range(self.V) if (i, j) in self.A) 
            #                                                         + gp.quicksum(f1[j, i, r, t_prime, t - 1] for j in range(self.V) for t_prime in range(t) 
            #                                                                       if (j, i) in self.A and (j, i, t_prime, t) in self.delta) 
            #                                                         - S[i, t] * g_coeff, name=f"InventoryUpdatePost_{i}_{r}_{t}")
                        
            # Inventory update post landfall
            inventory_update_post1= {}
            for i in range(self.V):
                for r in range(self.R):
                    for t in range(self.t_L + 1, self.T):
                        # Determine g_r coefficient
                        if i in self.V1:            
                            g_coeff = self.g_shelter[r] # shelter
                        elif i in self.V3:          
                            g_coeff = self.g_pod[r] # PoD
                        else:                  
                            g_coeff = 0 # warehouse or RSA
                        inventory_update_post1[i, r, t] = m.addConstr(I[i, r, t] - I[i, r, t - 1] + gp.quicksum(f[i, j, r, t] for j in range(self.V) if (i, j) in self.A) 
                                                                    - gp.quicksum(f[j, i, r, t_prime] for j in range(self.V) for t_prime in range(self.t_L, t) 
                                                                                  if (j, i) in self.A and (j, i, t_prime, t) in self.delta) 
                                                                    + S[i, t] * g_coeff <= 0, name=f"InventoryUpdatePost1_{i}_{r}_{t}")
                        
            inventory_update_post2= {}
            for i in range(self.V):
                for r in range(self.R):
                    for t in range(self.t_L + 1, self.T):
                        # Determine g_r coefficient
                        if i in self.V1:            
                            g_coeff = self.g_shelter[r] # shelter
                        elif i in self.V3:          
                            g_coeff = self.g_pod[r] # PoD
                        else:                  
                            g_coeff = 0 # warehouse or RSA
                        inventory_update_post2[i, r, t] = m.addConstr(I[i, r, t] - I[i, r, t - 1] + gp.quicksum(f[i, j, r, t] for j in range(self.V) if (i, j) in self.A) 
                                                                    - gp.quicksum(f[j, i, r, t_prime] for j in range(self.V) for t_prime in range(self.t_L, t) 
                                                                                  if (j, i) in self.A and (j, i, t_prime, t) in self.delta) 
                                                                    + S[i, t] * g_coeff >= 0, name=f"InventoryUpdatePost2_{i}_{r}_{t}")

        # Flow capacity post landfall
        flow_capacity_post = {}
        for (i, j) in self.A:
            for t in range(self.t_L, self.T):
                flow_capacity_post[i, j, t] = m.addConstr(gp.quicksum(self.w_r[r] * f[i, j, r, t] for r in range(self.R)) <= self.K_ijt[i][j][t], name=f"FlowCapacity_{i}_{j}_{t}")

        
        # if self.travel_time == 2: ***
        #     # In-transit flow init
        #     in_transit_update_constr1 = {}
        #     for (i, j) in self.A:
        #         for r in range(self.R):
        #             for t_prime in range(self.t_L): 
        #                 in_transit_update_constr1[i, j, r, t_prime] = m.addConstr(f1[i, j, r, t_prime, self.t_L] <= 0, name=f"CarryTransit1_{i}_{j}_{r}_{t_prime}_{self.t_L}")

        #     in_transit_update_constr2 = {}
        #     for (i, j) in self.A:
        #         for r in range(self.R):
        #             for t_prime in range(self.t_L): 
        #                 in_transit_update_constr2[i, j, r, t_prime] = m.addConstr(f1[i, j, r, t_prime, self.t_L] >= 0, name=f"CarryTransit2_{i}_{j}_{r}_{t_prime}_{self.t_L}")


        #     # In-transit flow post landfall
        #     in_transit_post = {}
        #     for (i, j) in self.A:
        #         for r in range(self.R):
        #             for t in range(self.t_L + 1, self.T):
        #                 for t_prime in range(t): 
        #                     if (i, j, t_prime, t) not in self.delta:
        #                         in_transit_post[i, j, r, t_prime, t] = m.addConstr(f1[i, j, r, t_prime, t] == f1[i, j, r, t_prime, t-1], name=f"CarryTransitPost_{i}_{j}_{r}_{t_prime}_{t}")
        #                     else:
        #                         in_transit_post[i, j, r, t_prime, t] = m.addConstr(f1[i, j, r, t_prime, t] == 0, name=f"CarryTransitPostZero{i}_{j}_{r}_{t_prime}_{t}")

        #     in_transit_post2 = {}
        #     for (i, j) in self.A:
        #         for r in range(self.R):
        #             for t in range(self.t_L, self.T):  # arrival period
        #                 # here t_prime == t
        #                 in_transit_post2[i, j, r, t] = m.addConstr(f1[i, j, r, t, t] == f[i, j, r, t], name=f"In_Transit_Post2_{i}_{j}_{r}_{t}")

        # Objective
        objective = (
            gp.quicksum(self.p_i[i] * U[i, t] for i in range(self.V) for t in range(self.t_L, self.T)) 
            + gp.quicksum(
                gp.quicksum(self.cost_ir[i][r] * f[i, j, r, t] for i in self.V0 for j in range(self.V) if (i, j) in self.A)
                + gp.quicksum(self.m_ij[i][j] * f[i, j, r, t] for (i, j) in self.A) 
                for r in range(self.R) for t in range(self.t_L, self.T)))
        
        m.setObjective(objective, GRB.MINIMIZE)
        m.update()
    
        if self.travel_time == 1:
            return m, I, U, S, f, demand_shortage_constr1, demand_shortage_constr2, demand_shortage_post, inventory_update_constr1, inventory_update_constr2, inventory_update_post, flow_capacity_post
        elif self.travel_time == 2:
            return m, I, U, S, f, demand_shortage_constr1, demand_shortage_constr2, demand_shortage_post, inventory_update_constr1, inventory_update_constr2, inventory_update_post1, inventory_update_post2, flow_capacity_post
    
    def build_last_stage(self):

        # Prepare dictionaries
        if self.travel_time == 1:
            self.demand_shortage_post = {}
            self.inventory_update_post = {}
            self.flow_capacity_post = {}
        elif self.travel_time == 2:
            self.demand_shortage_post = {}
            self.inventory_update_post1 = {}
            self.inventory_update_post2 = {}
            self.flow_capacity_post = {}
            # self.in_transit_post = {} ***
            # self.in_transit_post2 = {}

        for k in self.nodeLists[self.t_L]:
            if self.travel_time == 1:
                (
                    self.m[self.t_L, k],
                    self.I[self.t_L, k],
                    self.U[self.t_L, k],
                    self.S[self.t_L, k],
                    self.f[self.t_L, k],
                    self.demand_shortage_constr1[self.t_L, k],
                    self.demand_shortage_constr2[self.t_L, k],
                    self.demand_shortage_post[self.t_L, k],
                    self.inventory_update_constr1[self.t_L, k],
                    self.inventory_update_constr2[self.t_L, k],
                    self.inventory_update_post[self.t_L, k],
                    self.flow_capacity_post[self.t_L, k],
                ) = self.last_stage_model(k)
            elif self.travel_time == 2:
                (
                    self.m[self.t_L, k],
                    self.I[self.t_L, k],
                    self.U[self.t_L, k],
                    self.S[self.t_L, k],
                    self.f[self.t_L, k],
                    self.demand_shortage_constr1[self.t_L, k],
                    self.demand_shortage_constr2[self.t_L, k],
                    self.demand_shortage_post[self.t_L, k],
                    self.inventory_update_constr1[self.t_L, k],
                    self.inventory_update_constr2[self.t_L, k],
                    self.inventory_update_post1[self.t_L, k],
                    self.inventory_update_post2[self.t_L, k],
                    self.flow_capacity_post[self.t_L, k] #***
                ) = self.last_stage_model(k)

    def forward_pass(self):

        k_t = self.nodeLists[0][0]      # First state at t=0
        state_path = [k_t]              # Keep track of visited states
        # Initialize the state variables from t=0
        # Note that we need to define and record these for each stage as well, that's why we have an additional t index
        I_prev = np.zeros((self.V, self.R, self.T1))
        U_prev = np.zeros((self.V, self.T1))
        if self.travel_time == 2:
            f1_prev = {}
            for t in range(self.T1):
                for (i, j) in self.A:
                    for r in range(self.R):
                        for t_prime in range(t+1): 
                            f1_prev[i, j, r, t_prime, t] = 0.0 

        thetaval = [0] * self.T1     # Stores theta values (cost-to-go)
        lb = self.lower_bound

        for t in range(self.T1):
            if t > 0:
                k_t = msp_helper.MC_sample(t - 1, k_t, self.transition_dict)
                state_path.append(k_t)          # Keep sample path history
                if self.travel_time == 1:
                    self.update_rhs_time1(t, k_t, I_prev, U_prev)         
                elif self.travel_time == 2:
                    self.update_rhs(t, k_t, I_prev, U_prev, f1_prev) # Update the right hand side of the constraints
            self.m[t, k_t].optimize() # Optimize the model
            # if t == self.t_L:
            #     self.m[t, k_t].write(f"LP/last_stage_model_t{t}_k{k_t}.lp")

            # # Debugging
            # print(f"\n--- Stage {t}, State {k_t} ---")

            # # Print Inventory (I)
            # for i in range(self.V):
            #     for r in range(self.R):
            #         if t < self.t_L:
            #             val = self.I[t, k_t][i, r].x
            #             if val > 1e-6:  
            #                 print(f"I[{i},{r}] = {val}")
            #         else:
            #             for t_post in range(self.t_L, self.T):
            #                 val = self.I[self.t_L, k_t][i, r, t_post].x
            #                 if val > 1e-6:
            #                     print(f"I[{i},{r},{t_post}] = {val}")

            # # Print Unmet Demand (U)
            # for i in range(self.V):
            #     if t < self.t_L:
            #         val = self.U[t, k_t][i].x
            #         if val > 1e-6:
            #             print(f"U[{i}] = {val}")
            #     else:
            #         for t_post in range(self.t_L, self.T):
            #                 val = self.U[self.t_L, k_t][i, t_post].x
            #                 if val > 1e-6:
            #                     print(f"U[{i},{t_post}] = {val}")

            # # Print Satisfied Demand (S)
            # for i in range(self.V):
            #     if t < self.t_L:
            #         val = self.S[t, k_t][i].x
            #         if val > 1e-6:
            #             print(f"S[{i}] = {val}")
            #     else:
            #         for t_post in range(self.t_L, self.T):
            #             val = self.S[self.t_L, k_t][i, t_post].x
            #             if val > 1e-6:
            #                 print(f"S[{i},{t_post}] = {val}")

            # # Print Local Flow (f)
            # for i in range(self.V):
            #     for j in range(self.V):
            #         if (i, j) in self.A:
            #             for r in range(self.R):
            #                 if t < self.t_L:
            #                     val = self.f[t, k_t][i, j, r].x
            #                     if val > 1e-6:
            #                         print(f"f[{t}][{i},{j},{r}] = {val}")
            #                 else:
            #                     for t_post in range(self.t_L, self.T):
            #                         val = self.f[self.t_L, k_t][i, j, r, t_post].x
            #                         if val > 1e-6:
            #                             print(f"f[{i},{j},{r},{t_post}] = {val}")

            # # Print In-Transit Flows (f1)
            # if self.travel_time == 2:
            #     for i in range(self.V):
            #         for j in range(self.V):
            #             if (i, j) in self.A:
            #                 for r in range(self.R):
            #                     if t < self.t_L:
            #                         for t_prime in range(t+1):
            #                             val = self.f1[t, k_t][i, j, r, t_prime].x
            #                             if val > 1e-6:
            #                                 print(f"f1[{t}][{i},{j},{r},{t_prime}] = {val}")
            #                     else:
            #                         for t_post in range(self.t_L, self.T):
            #                             for t_prime in range(t_post+1):
            #                                 val = self.f1[self.t_L, k_t][i, j, r, t_prime, t_post].x
            #                                 if val > 1e-6:
            #                                     print(f"f1[{t}][{i},{j},{r},{t_prime},{t_post}] = {val}")

            # # Print Theta
            # print(f"Theta: {self.theta[t, k_t].x}")

            # Check for the optimality and then extract the solution for updating state variables
            if self.m[t, k_t].status != GRB.OPTIMAL:
                print("Error in Forward Pass")
                print(f"Model in stage = {t} and state = {k_t}, in forward pass is {self.m[t, k_t].status}")
                sys.exit(0)
            else:
                if self.last_stage == 1:
                    for i in range(self.V):
                        for r in range(self.R):
                            I_prev[i, r, t] = self.I[t, k_t][i, r].x # Extract inventory level (state var)
                        U_prev[i, t] = self.U[t, k_t][i].x # Extract demand shortage (state var)

                    if self.travel_time == 2: 
                        for (i, j) in self.A:
                            for r in range(self.R):
                                for t_prime in range(t+1):  # Only t′ < t matters
                                    f1_prev[i, j, r, t_prime, t] = self.f1[t, k_t][i, j, r, t_prime].x

                    thetaval[t] = self.theta[t, k_t].x

                elif self.last_stage == 2:
                    if t < self.t_L:
                        for i in range(self.V):
                            for r in range(self.R):
                                I_prev[i, r, t] = self.I[t, k_t][i, r].x # Extract inventory level (state var)
                            U_prev[i, t] = self.U[t, k_t][i].x # Extract demand shortage (state var)

                        if self.travel_time == 2: 
                            for (i, j) in self.A:
                                for r in range(self.R):
                                    for t_prime in range(t+1): 
                                        f1_prev[i, j, r, t_prime, t] = self.f1[t, k_t][i, j, r, t_prime].x

                        thetaval[t] = self.theta[t, k_t].x

                if t == 0:
                    lb = self.m[t, k_t].objVal            

            # # Capture f_hat values # ***
            # if self.travel_time == 1:
            #     if t in [self.t_L - 2, self.t_L - 1]:
            #         for j in range(self.V):
            #             for i in range(self.V):
            #                 if (j, i) in self.A:
            #                     for r in range(self.R):
            #                         self.f_hat[j, i, r, t] = self.f[t, k_t][j, i, r].x
            #                         # if self.f_hat[j, i, r, t] > 1e-6:
            #                         #     print(f"f_hat[{j},{i},{r},{t}] = {self.f_hat[j, i, r, t]:.2f}")

        # print("state path:", state_path)
        # print("last stage objective function:", self.m[self.t_L, k_t].objVal)
        
        if self.travel_time == 1:
            return I_prev, U_prev, thetaval, lb, state_path
        elif self.travel_time == 2:
            return I_prev, U_prev, f1_prev, thetaval, lb, state_path

    def update_rhs(self, t, k_t, I_prev, U_prev, f1_prev):
        """Update right-hand sides of the (t , k_t) model constraints before optimization.""" 

        # Realized demand at stage t, given Markov state k_t
        demand_real = self.demand_processor.msp_demand_mapping(t, k_t, self.network_processor)
        
        # print(f"\n--- Demand at stage t = {t}, state = {k_t} ---")
        # for i in range(self.V):
        #     if demand_real[i] > 1e-6:
        #         print(f"Demand at node {i} = {demand_real[i]}")

        # Demand shortage rhs
        for i in range(self.V):  # Loop over all locations/nodes
            self.demand_shortage_constr1[t, k_t][i].setAttr(GRB.Attr.RHS, U_prev[i, t - 1] + demand_real[i])
            self.demand_shortage_constr2[t, k_t][i].setAttr(GRB.Attr.RHS, U_prev[i, t - 1] + demand_real[i])

        # Inventory flow-balance rhs
        # We are at stage t and updating rhs for constraints in stage t, so we check the condition at this stage but add the state variables from previous stage.
            for r in range(self.R):         
                rhs_val = I_prev[i, r, t - 1] + sum(f1_prev[j, i, r, t_prime, t - 1] for j in range(self.V) for t_prime in range(t) if (j, i) in self.A and (j, i, t_prime, t) in self.delta)
                self.inventory_update_constr1[t, k_t][i, r].setAttr(GRB.Attr.RHS, rhs_val)
                self.inventory_update_constr2[t, k_t][i, r].setAttr(GRB.Attr.RHS, rhs_val)

        # Inventory flow-balance post rhs ***
                if t == self.t_L:
                    for tt in range(self.t_L + 1, self.T):
                        self.inventory_update_post1[t, k_t][i, r, tt].setAttr(GRB.Attr.RHS, sum(f1_prev[j, i, r, t_prime, t - 1] for j in range(self.V) for t_prime in range(self.t_L) if (j, i) in self.A and (j, i, t_prime, tt) in self.delta))
                        self.inventory_update_post2[t, k_t][i, r, tt].setAttr(GRB.Attr.RHS, sum(f1_prev[j, i, r, t_prime, t - 1] for j in range(self.V) for t_prime in range(self.t_L) if (j, i) in self.A and (j, i, t_prime, tt) in self.delta))

        # In-transit flow rhs ***
        if t < self.t_L:
            for (i, j) in self.A:
                for r in range(self.R):
                    for t_prime in range(t):  # Only for past departure times
                        if (i, j, t_prime, t) not in self.delta: # Check whether flow (i, j) started at t′ is still in transit at stage t
                            rhs_val = f1_prev[i, j, r, t_prime, t - 1] # If it hasn't arrived yet, carry forward from previous stage
                        else:
                            rhs_val = 0.0 # If it has arrived, flow is no longer in transit
                        self.in_transit_update_constr1[t, k_t][i, j, r, t_prime].setAttr(GRB.Attr.RHS, rhs_val)
                        self.in_transit_update_constr2[t, k_t][i, j, r, t_prime].setAttr(GRB.Attr.RHS, rhs_val)
    
    """
    def update_rhs_time1(self, t, k_t, I_prev, U_prev):
        # Update right-hand sides of the (t , k_t) model constraints before optimization.
        
        # Realized demand at stage t, given Markov state k_t
        demand_real = self.demand_processor.msp_demand_mapping(t, k_t, self.network_processor)
        
        # Demand shortage rhs
        for i in range(self.V):  # Loop over all locations/nodes
            self.demand_shortage_constr1[t, k_t][i].setAttr(GRB.Attr.RHS, U_prev[i, t - 1] + demand_real[i])
            self.demand_shortage_constr2[t, k_t][i].setAttr(GRB.Attr.RHS, U_prev[i, t - 1] + demand_real[i])

        # Inventory flow-balance rhs
        for i in range(self.V):
            for r in range(self.R):
                self.inventory_update_constr1[t, k_t][i, r].setAttr(GRB.Attr.RHS, I_prev[i, r, t - 1])
                self.inventory_update_constr2[t, k_t][i, r].setAttr(GRB.Attr.RHS, I_prev[i, r, t - 1])
    """
    def backward_pass(self, I_prev, U_prev, f1_prev, thetaval, state_path):
        
        cutviolFlag = False

        for t in range(len(state_path)-1, 0, -1):
            Q = {}  # Use dictionary to store: key = Markov state (tuple), value = objVal
            # List for all the dual multipliers of the constraints
            dual_demand1    = {}  # (k,i)     
            dual_demand2    = {}  # (k,i)          
            dual_inventory1 = {}  # (k,i,r)   
            dual_inventory2 = {}  # (k,i,r)
            dual_inventory_post1 = {} # (k, i, r, tt)
            dual_inventory_post2 = {} # (k, i, r, tt)
            dual_transit1    = {}  # (k,i,j,r,t′) ***
            dual_transit2    = {}  # (k,i,j,r,t′)    

            sample_state = state_path[t - 1]  # The state observed at time t-1

            for k_state in self.nodeLists[t]:
                self.update_rhs(t, k_state, I_prev, U_prev, f1_prev)
                self.m[t, k_state].optimize()
                if self.m[t, k_state].status != gp.GRB.OPTIMAL:
                    print("Error in Backward Pass")
                    print(f"Model in stage = {t} and state = {k_state}, in backward pass is {self.m[t, k_state].status}")
                    sys.exit(0)
                else:
                    Q[k_state] = self.m[t, k_state].ObjVal    # Save the objective value for this (t , k_state)

                    # Extract dual values for each constraint set
                    for i in range(self.V):
                        dual_demand1[k_state, i] = self.demand_shortage_constr1[t, k_state][i].pi 
                        dual_demand2[k_state, i] = self.demand_shortage_constr2[t, k_state][i].pi 

                        for r in range(self.R):
                            dual_inventory1[k_state, i, r] = self.inventory_update_constr1[t, k_state][i, r].pi
                            dual_inventory2[k_state, i, r] = self.inventory_update_constr2[t, k_state][i, r].pi 

                            if t == self.t_L: # ***
                                for tt in range(self.t_L + 1, self.T):     
                                    dual_inventory_post1[k_state, i, r, tt] = self.inventory_update_post1[t, k_state][i, r, tt].pi
                                    dual_inventory_post2[k_state, i, r, tt] = self.inventory_update_post2[t, k_state][i, r, tt].pi

                    if t < self.t_L:
                        for (i, j) in self.A: #***
                            for r in range(self.R):
                                for t_prime in range(t):
                                    dual_transit1[k_state, i, j, r, t_prime] = self.in_transit_update_constr1[t, k_state][i, j, r, t_prime].pi
                                    dual_transit2[k_state, i, j, r, t_prime] = self.in_transit_update_constr2[t, k_state][i, j, r, t_prime].pi

            # Solving all stage-(t-1) problems and generate cuts/valid inequalities
            for n in self.nodeLists[t - 1]:
                Qvalue = 0
                for k_state in self.nodeLists[t]:
                    # Accumulate expected cost-to-go from all reachable next-stage states with non-negligible transition probability
                    if self.transition_dict[str(t-1)][str(n)][str(k_state)] > self.smallest_trans_prob:
                        Qvalue += Q[k_state] * self.transition_dict[str(t-1)][str(n)][str(k_state)]

                # Check if cut is violated at the sample path encountered in the forward pass (Only interested in checking cut violation on the same node as the forward pass)
                # n  : current state at stage t-1   # sample_state : state that was actually visited in the forward pass
                if n == sample_state and (
                        (Qvalue - thetaval[t - 1]) / max(1e-10, abs(thetaval[t - 1])) > self.cutviol 
                        and abs(Qvalue - thetaval[t - 1]) > self.cutviol
                ):
                    cutviolFlag = True # Note that if this is True we generate the cut not only for this state, but also for all other states at the relevant state.

                # Initialize coeff-lists for each state variable
                cutcoef_U1 = [0] * self.V
                cutcoef_U2 = [0] * self.V
                cutcoef_I1 = [[0] * self.R for _ in range(self.V)]  
                cutcoef_I2 = [[0] * self.R for _ in range(self.V)]
                if t < self.t_L:
                    cutcoef_f1 = {(i, j, r, t_prime): 0.0 for (i, j) in self.A for r in range(self.R) for t_prime in range(t)} #***     
                    cutcoef_f2 = {(i, j, r, t_prime): 0.0 for (i, j) in self.A for r in range(self.R) for t_prime in range(t)} 
                else: # ***
                    cutcoef_I3 = [[[0] * (self.T - (self.t_L + 1)) for r in range(self.R)] for i in range(self.V)]
                    cutcoef_I4 = [[[0] * (self.T - (self.t_L + 1)) for r in range(self.R)] for i in range(self.V)]
                cut_rhs = 0.0   

                for k_state in self.nodeLists[t]:
                    if self.transition_dict[str(t-1)][str(n)][str(k_state)] > self.smallest_trans_prob:
                        
                        # No matter what are the sign of dual variables, we "add" them for the coefficients and the rhs both. 
                        # Demand-shortage constraints
                        for i in range(self.V):
                            lam1 = dual_demand1[k_state, i] * self.transition_dict[str(t-1)][str(n)][str(k_state)] 
                            cutcoef_U1[i] += lam1                                                 
                            cut_rhs += lam1 * U_prev[i, t-1]                                          

                            lam2 = dual_demand2[k_state, i] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                            cutcoef_U2[i] += lam2                                                
                            cut_rhs += lam2 * U_prev[i, t-1]                                           

                        # Inventory-balance constraints
                        # State variable I
                            for r in range(self.R):
                                lam1 = dual_inventory1[k_state, i, r] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                cutcoef_I1[i][r] += lam1
                                summation = sum(f1_prev[j, i, r, t_prime, t - 1] for j in range(self.V) for t_prime in range(t) if (j, i) in self.A and (j, i, t_prime, t) in self.delta)                                          
                                cut_rhs += lam1 * (I_prev[i, r, t-1] + summation)                                   

                                lam2 = dual_inventory2[k_state, i, r] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                cutcoef_I2[i][r] += lam2                                          
                                cut_rhs += lam2 * (I_prev[i, r, t-1] + summation) 

                                # Inventory-balance post constraints ***
                                if t == self.t_L:
                                    for tt in range(self.t_L + 1, self.T):
                                        lam1 = dual_inventory_post1[k_state, i, r, tt] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                        cutcoef_I3[i][r][tt - (self.t_L + 1)] += lam1
                                        cut_rhs += lam1 * sum(f1_prev[j, i, r, t_prime, t - 1] for j in range(self.V) for t_prime in range(self.t_L) if (j, i) in self.A and (j, i, t_prime, tt) in self.delta)

                                        lam2 = dual_inventory_post2[k_state, i, r, tt] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                        cutcoef_I4[i][r][tt - (self.t_L + 1)] += lam2
                                        cut_rhs += lam2 * sum(f1_prev[j, i, r, t_prime, t - 1] for j in range(self.V) for t_prime in range(self.t_L) if (j, i) in self.A and (j, i, t_prime, tt) in self.delta)

                        # In-transit flow constraints ***
                        if t < self.t_L:
                            for (i, j) in self.A:
                                for r in range(self.R):
                                    for t_prime in range(t):
                                        if (i, j, t_prime, t) not in self.delta:
                                            lam1 = dual_transit1[k_state, i, j, r, t_prime] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                            cutcoef_f1[(i, j, r, t_prime)] += lam1 
                                            cut_rhs += lam1 * f1_prev[i, j, r, t_prime, t-1]

                                            lam2 = dual_transit2[k_state, i, j, r, t_prime] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                            cutcoef_f2[(i, j, r, t_prime)] += lam2 
                                            cut_rhs += lam2 * f1_prev[i, j, r, t_prime, t-1]

                # Add the cut
                # Pay attention to the signs of the terms on the left ahdn side and sign of cut_rhs on thr right hand side 
                if t == self.t_L: #***
                    self.m[t-1, n].addConstr(self.theta[t-1, n]
                                        - gp.quicksum(cutcoef_U1[i] * self.U[t-1, n][i] for i in range(self.V)) 
                                        - gp.quicksum(cutcoef_U2[i] * self.U[t-1, n][i] for i in range(self.V))
                                        - gp.quicksum(cutcoef_I1[i][r] * self.I[t-1, n][i, r] for i in range(self.V) for r in range(self.R))
                                        - gp.quicksum(cutcoef_I2[i][r] * self.I[t-1, n][i, r] for i in range(self.V) for r in range(self.R))
                                        - gp.quicksum(cutcoef_I1[i][r] * gp.quicksum(self.f1[t-1, n][j, i, r, t_prime] for j in range(self.V) for t_prime in range(t) if (j, i) in self.A and (j, i, t_prime, t) in self.delta) for i in range(self.V) for r in range(self.R))
                                        - gp.quicksum(cutcoef_I2[i][r] * gp.quicksum(self.f1[t-1, n][j, i, r, t_prime] for j in range(self.V) for t_prime in range(t) if (j, i) in self.A and (j, i, t_prime, t) in self.delta) for i in range(self.V) for r in range(self.R))
                                        - gp.quicksum(cutcoef_I3[i][r][tt - (self.t_L + 1)] * gp.quicksum(self.f1[t-1, n][j, i, r, t_prime] for j in range(self.V) for t_prime in range(self.t_L) if (j, i) in self.A and (j, i, t_prime, tt) in self.delta) for i in range(self.V) for r in range(self.R) for tt in range(self.t_L + 1, self.T))
                                        - gp.quicksum(cutcoef_I4[i][r][tt - (self.t_L + 1)] * gp.quicksum(self.f1[t-1, n][j, i, r, t_prime] for j in range(self.V) for t_prime in range(self.t_L) if (j, i) in self.A and (j, i, t_prime, tt) in self.delta) for i in range(self.V) for r in range(self.R) for tt in range(self.t_L + 1, self.T))
                                        >= Qvalue - cut_rhs, name=f"OptCut_{t-1}_{n}")
                else:
                    self.m[t-1, n].addConstr(self.theta[t-1, n] 
                                        - gp.quicksum(cutcoef_U1[i] * self.U[t-1, n][i] for i in range(self.V)) 
                                        - gp.quicksum(cutcoef_U2[i] * self.U[t-1, n][i] for i in range(self.V))
                                        - gp.quicksum(cutcoef_I1[i][r] * self.I[t-1, n][i, r] for i in range(self.V) for r in range(self.R))
                                        - gp.quicksum(cutcoef_I2[i][r] * self.I[t-1, n][i, r] for i in range(self.V) for r in range(self.R))
                                        - gp.quicksum(cutcoef_I1[i][r] * gp.quicksum(self.f1[t-1, n][j, i, r, t_prime] for j in range(self.V) for t_prime in range(t) if (j, i) in self.A and (j, i, t_prime, t) in self.delta) for i in range(self.V) for r in range(self.R))
                                        - gp.quicksum(cutcoef_I2[i][r] * gp.quicksum(self.f1[t-1, n][j, i, r, t_prime] for j in range(self.V) for t_prime in range(t) if (j, i) in self.A and (j, i, t_prime, t) in self.delta) for i in range(self.V) for r in range(self.R))
                                        - gp.quicksum(cutcoef_f1[(i, j, r, t_prime)] * self.f1[t-1, n][i, j, r, t_prime] for (i, j) in self.A for r in range(self.R) for t_prime in range(t)) 
                                        - gp.quicksum(cutcoef_f2[(i, j, r, t_prime)] * self.f1[t-1, n][i, j, r, t_prime] for (i, j) in self.A for r in range(self.R) for t_prime in range(t))
                                        >= Qvalue - cut_rhs, name=f"OptCut_{t-1}_{n}")
                
                # if t == 9:
                #     self.m[t-1, n].update()
                #     self.m[t-1, n].write(f"LP/first_stage_model_t{t-1}_n{n}.lp")

        return cutviolFlag
    """
    def backward_pass_time1(self, I_prev, U_prev, thetaval, state_path):
                
        cutviolFlag = False

        for t in range(len(state_path)-1, 0, -1):

            Q = {}  # Use dictionary to store: key = Markov state (tuple), value = objVal
            # List for all the dual multipliers of the constraints
            dual_demand1    = {}  # (k,i)     
            dual_demand2    = {}  # (k,i)          
            dual_inventory1 = {}  # (k,i,r)   
            dual_inventory2 = {}  # (k,i,r)                   

            sample_state = state_path[t - 1]  # The state observed at time t-1

            for k_state in self.nodeLists[t]:
                self.update_rhs_time1(t, k_state, I_prev, U_prev)
                self.m[t, k_state].optimize()
                if self.m[t, k_state].status != gp.GRB.OPTIMAL:
                    print("Error in Backward Pass")
                    print(f"Model in stage = {t} and state = {k_state}, in backward pass is {self.m[t, k_state].status}")
                    sys.exit(0)
                else:
                    Q[k_state] = self.m[t, k_state].objVal    # Save the objective value for this (t , k_state)
 
                    for i in range(self.V):
                        dual_demand1[k_state, i] = self.demand_shortage_constr1[t, k_state][i].pi # <= 0 negative

                    for i in range(self.V):
                        dual_demand2[k_state, i] = self.demand_shortage_constr2[t, k_state][i].pi # >= 0 positive

                    for i in range(self.V):
                        for r in range(self.R):
                            dual_inventory1[k_state, i, r] = self.inventory_update_constr1[t, k_state][i, r].pi # <= 0 negative
                
                    for i in range(self.V):
                        for r in range(self.R):
                            dual_inventory2[k_state, i, r] = self.inventory_update_constr2[t, k_state][i, r].pi # >= 0 positive
                            
            # Solving all stage-(t-1) problems and generate cuts/valid inequalities
            for n in self.nodeLists[t - 1]:
                Qvalue = 0
                for k_state in self.nodeLists[t]:
                    # Accumulate expected cost-to-go from all reachable next-stage states with non-negligible transition probability
                    if self.transition_dict[str(t-1)][str(n)][str(k_state)] > self.smallest_trans_prob:
                        Qvalue += Q[k_state] * self.transition_dict[str(t-1)][str(n)][str(k_state)]

                # Check if cut is violated at the sample path encountered in the forward pass (Only interested in checking cut violation on the same node as the forward pass)
                # n  : current state at stage t-1   # sample_state : state that was actually visited in the forward pass
                if n == sample_state and (
                        (Qvalue - thetaval[t - 1]) / max(1e-10, abs(thetaval[t - 1])) > self.cutviol 
                        and abs(Qvalue - thetaval[t - 1]) > self.cutviol
                ):
                    cutviolFlag = True # Note that if this is True we generate the cut not only for this state, but also for all other states at the relevant state.

                # Initialize coeff-dictionaries for each state variable
                cutcoef_U1 = [0] * self.V
                cutcoef_U2 = [0] * self.V
                cutcoef_I1 = [[0] * self.R for _ in range(self.V)]  
                cutcoef_I2 = [[0] * self.R for _ in range(self.V)]            
                cut_rhs = 0.0   

                for k_state in self.nodeLists[t]:
                    if self.transition_dict[str(t-1)][str(n)][str(k_state)] > self.smallest_trans_prob:
                        
                        # No matter what are the sign of dual variables, we "add" them for the coefficients and the rhs both. 
                        # Demand-shortage constraints
                        for i in range(self.V):
                            lam1 = dual_demand1[k_state, i] * self.transition_dict[str(t-1)][str(n)][str(k_state)] 
                            cutcoef_U1[i] += lam1                                                 # Coefficient
                            cut_rhs += lam1 * U_prev[i][t-1]                                          # RHS

                            lam2 = dual_demand2[k_state, i] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                            cutcoef_U2[i] += lam2                                                 # Coefficient
                            cut_rhs += lam2 * U_prev[i][t-1]                                           # RHS

                        # Inventory-balance constraints
                        for i in range(self.V):
                            for r in range(self.R):
                                lam1 = dual_inventory1[k_state, i, r] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                cutcoef_I1[i][r] += lam1                                          # Coefficient
                                cut_rhs += lam1 * I_prev[i][r][t-1]                                   # RHS

                                lam2 = dual_inventory2[k_state, i, r] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                cutcoef_I2[i][r] += lam2                                          # Coefficient
                                cut_rhs += lam2 * I_prev[i][r][t-1]                                    # RHS
              
                # Add the cut
                # Pay attention to the signs of the terms on the left ahdn side and sign of cut_rhs on thr right hand side          
                self.m[t-1, n].addConstr(self.theta[t-1, n]
                                        - gp.quicksum(cutcoef_U1[i] * self.U[t-1, n][i] for i in range(self.V)) 
                                        - gp.quicksum(cutcoef_U2[i] * self.U[t-1, n][i] for i in range(self.V))
                                        - gp.quicksum(cutcoef_I1[i][r] * self.I[t-1, n][i, r] for i in range(self.V) for r in range(self.R))
                                        - gp.quicksum(cutcoef_I2[i][r] * self.I[t-1, n][i, r] for i in range(self.V) for r in range(self.R)) 
                                        >= Qvalue - cut_rhs, name=f"OptCut_{t-1}_{n}")
               
        return cutviolFlag
    """
    # Train model
    def solve_sddp(self):

        self.build_models()
        if self.last_stage == 2:
            self.build_last_stage()

        I_ir0 = self.network_processor.I_ir0
        k_init = self.nodeLists[0][0]

        # Set the RHS of the first_stage problem
        for i in range(self.V):
            for r in range(self.R):
                self.inventory_update_constr1[0, k_init][i, r].setAttr(GRB.Attr.RHS, I_ir0[i][r])
        
        for i in range(self.V):
            for r in range(self.R):
                self.inventory_update_constr2[0, k_init][i, r].setAttr(GRB.Attr.RHS, I_ir0[i][r])

        # Initialization
        train_time = 0
        relative_gap = self.rel_gap
        lb = 0
        LB = []  # lower-bound history 
        thetaval = [0] * self.T1
        iter = 0
        cutviol_iter = 0
        start = time.time()

        while True:
            iter += 1

            # Forward pass
            if self.travel_time == 1:
                I_prev, U_prev, thetaval, lb, state_path = self.forward_pass()
            elif self.travel_time == 2:
                I_prev, U_prev, f1_prev, thetaval, lb, state_path = self.forward_pass()
            # exit(0)
            
            LB.append(lb)

            # Termination check
            flag, Elapsed = msp_helper.termination_check(iter, relative_gap, LB, start, cutviol_iter, self.params)
            if flag != 0:
                train_time = Elapsed
                break
            
            # Backward pass (if not terminated)
            if self.travel_time == 1:
                cut_violFlag = self.backward_pass_time1(I_prev, U_prev, thetaval, state_path)
            elif self.travel_time == 2:
                cut_violFlag = self.backward_pass(I_prev, U_prev, f1_prev, thetaval, state_path)

            # exit(0)

            # cut-violation
            if cut_violFlag:
                cutviol_iter = 0
            else:
                cutviol_iter += 1
            
            print(f"Iter {iter}: LB = {lb:.2f}, Cut Violated = {cut_violFlag}")

            # Save LB history to CSV
            import csv
            output_path = "SDDP_LB_history.csv"
            with open(output_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Iteration", "LB"])
                for i, lb_val in enumerate(LB, start=1):
                    writer.writerow([i, lb_val])

        return LB, train_time, iter
    
    # Evaluate model
    def sddp_eval(self, sample_paths_file, demand_processor):
        
        if self.last_stage == 1:
            nbOS = demand_processor.nbOS

            # Load all sample paths from CSV
            OS_paths_df = pd.read_csv(sample_paths_file)
            OS_paths = OS_paths_df.values[:nbOS] # Converts the DataFrame into a NumPy array

            objs = np.zeros((nbOS, self.T1))
            penalty_cost = np.zeros(nbOS)
            proc_cost = np.zeros(nbOS)
            trans_cost = np.zeros(nbOS)
            unmet_amount = np.zeros(nbOS)
            satisfied_amount = np.zeros(nbOS)
            proc_amount = np.zeros(nbOS)
            trans_amount = np.zeros(nbOS)
            Uval = {}
            fval = {}

            start = time.time()

            # For heatmap
            # Shape: (6 intensity [0–5], T1–1 time steps excluding t=0)
            inventory_sum = np.zeros((6, self.T1 - 1)) # An array holding the sum of I values (per intensity per time) across all sample paths. We don't consider t=0 because we start from one common state.
            unmet_sum = np.zeros((6, self.T1 - 1)) # An array holding the sum of U values (per intensity per time) across all sample paths.
            satisfied_sum = np.zeros((6, self.T1 - 1)) # An array holding the sum of S values (per intensity per time) across all sample paths
            visit_count = np.zeros((6, self.T1 - 1)) # An array to count how many sample paths visit each (intensity, time) pair 
            
            for s in range(nbOS):
                I_prev = np.zeros((self.V, self.R, self.T1))
                U_prev = np.zeros((self.V, self.T1))
                if self.travel_time == 2:
                    f1_prev = np.zeros((self.V, self.V, self.R, self.T1, self.T1))
                for t in range(self.T1):
                    k_t = eval(OS_paths[s, t]) 
                    if t > 0:
                        intensity = k_t[1]  # extract the intensity from (track_error, intensity)
                        visit_count[intensity][t - 1] += 1 # Update count
                        if self.travel_time == 1:
                            self.update_rhs_time1(t, k_t, I_prev, U_prev)
                        elif self.travel_time == 2:
                            self.update_rhs(t, k_t, I_prev, U_prev, f1_prev)                
                    self.m[t, k_t].optimize()
                    if self.m[t, k_t].status != GRB.OPTIMAL:
                        print(" in evaluation")
                        print(f"Model in stage = {t} and state = {k_t}, in forward pass is {self.m[t,k_t].status}")
                        exit(0)
                    else:
                        objs[s, t] = self.m[t,k_t].ObjVal - self.theta[t, k_t].x
                        Uval[s, t] = {}
                        fval[s, t] = {}
                        for i in range(self.V):
                            for r in range(self.R):
                                I_prev[i, r, t] = self.I[t, k_t][i, r].x # Extract inventory level (state var)

                                if t > 0 and i in self.V1: # Only include inventory at demand locations before landfall (shelters) for t > 0
                                    inventory_sum[intensity][t - 1] += I_prev[i, r, t] # For heat map
 
                            U_prev[i, t] = self.U[t, k_t][i].x
                            unmet_amount[s] += U_prev[i, t] # Accumulate amount of unmet demand before landfall
                            satisfied_amount[s] += self.S[t, k_t][i].x # Accumulate amount of satisfied demand before landfall
                            Uval[s, t][i] = self.U[t, k_t][i].x
                            if t > 0 and i in self.V1: # Only include inventory at demand locations before landfall (shelters) for t > 0
                                unmet_sum[intensity][t - 1] += U_prev[i, t] # For heat map
                                satisfied_sum[intensity][t - 1] += self.S[t, k_t][i].x # For heat map
                        
                        if self.travel_time == 2:
                            for i in range(self.V):
                                for j in range(self.V):
                                    if (i, j) in self.A:
                                        for r in range(self.R):
                                            for t_prime in range(t+1):
                                                f1_prev[i, j, r, t_prime, t] = self.f1[t, k_t][i, j, r, t_prime].x                                      
                        
                        # Computing procurement and transportation cost
                        for i in range(self.V):
                            for j in range(self.V):
                                if (i, j) in self.A:
                                    for r in range(self.R):
                                        fval[s, t][i, j, r] = self.f[t, k_t][i, j, r].x
                                        if i in self.V0:  # Procurement cost only from warehouses
                                            proc_cost[s] += self.cost_ir[i][r] * fval[s, t][i, j, r] 
                                            proc_amount[s] += fval[s, t][i, j, r]                     
                                        trans_cost[s] += self.m_ij[i][j] * fval[s, t][i, j, r]
                                        trans_amount[s] += fval[s, t][i, j, r]

                    # Computing penalty cost for each sample path
                    for i in self.V1:
                        penalty_cost[s] += self.p_i[i] * Uval[s, t][i]                    

            # Printing amount of variables
            print("Average amount of unmet demand=", np.mean(unmet_amount))
            print("Average amount of satisfied demand=", np.mean(satisfied_amount))
            print("Average amount of procurement=", np.mean(proc_amount))
            print("Average amount of transportation=", np.mean(trans_amount))
            
            # Printing costs
            print("Average Penalty Cost =", np.mean(penalty_cost))
            print("Average Procurement Cost =", np.mean(proc_cost))
            print("Average Transportation Cost =", np.mean(trans_cost))

            mean_obj = np.mean(np.sum(objs, axis=1))
            std_obj = np.std(np.sum(objs, axis=1))
            ci_low = mean_obj - 1.96 * std_obj / np.sqrt(nbOS)
            ci_high = mean_obj + 1.96 * std_obj / np.sqrt(nbOS)
            CI = mean_obj - ci_low
            print(f"\nSDDP Evaluation Results")
            print(f"μ ± 1.96*σ/√NS = {mean_obj} ± {CI}")
            print(f"95% Confidence Interval = [{ci_low:.2f}, {ci_high:.2f}]")
            test_time = time.time() - start
            
            # for t in range(self.T1 - 1):  # time steps 1 to T1-1
            #     print(f"\nTime step t = {t + 1}:")
            #     for i in range(6):  # intensity levels 0 to 5
            #         print(f"  Intensity {i}: {int(visit_count[i][t])} paths")

            # ========================= Heat map ==============================
            # Computing average inventory per intensity per stage
            avg_inventory = np.divide(inventory_sum, visit_count, out=np.full_like(inventory_sum, np.nan), where=visit_count != 0)
            # print("Inventory:\n", avg_inventory)
            avg_unmet = np.divide(unmet_sum, visit_count, out=np.full_like(unmet_sum, np.nan), where=visit_count != 0)
            # print("Unmet:\n", avg_unmet)
            avg_satisfied = np.divide(satisfied_sum, visit_count, out=np.full_like(satisfied_sum, np.nan), where=visit_count != 0)
            # print("Satisfied:\n", avg_satisfied)
            
            # Plotting inventory heat map
            plt.figure(figsize=(6, 6))
            sns.heatmap(
                avg_inventory,               # your masked average inventory matrix
                cmap="viridis",              
                cbar_kws={'label': 'Average Inventory'},
                xticklabels=range(1, avg_inventory.shape[1] + 1),  # time steps: t = 1 to T1 - 1
                yticklabels=range(avg_inventory.shape[0]),         # intensity: 0 to 5
                linewidths=0.3,
                linecolor='gray',
                square=True,
                mask=np.isnan(avg_inventory)  # this is key: white-out the unreachable states
            )
            plt.title("Inventory Heatmap Over Time by Intensity")
            plt.xlabel("Time")
            plt.ylabel("Intensity")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            save_path = os.path.join("Heat map", "inventory_M1=4_M2=1.png")
            plt.savefig(save_path, dpi=300)
            print(f"Heatmap saved to: {save_path}")
            
            # Plotting unmet demand heat map
            plt.figure(figsize=(6, 6))
            sns.heatmap(
                avg_unmet,               # your masked average unmet matrix
                cmap="viridis",              
                cbar_kws={'label': 'Average Unmet Demand'},
                xticklabels=range(1, avg_unmet.shape[1] + 1),  # time steps: t = 1 to T1 - 1
                yticklabels=range(avg_unmet.shape[0]),         # intensity: 0 to 5
                linewidths=0.3,
                linecolor='gray',
                square=True,
                mask=np.isnan(avg_unmet)  # this is key: white-out the unreachable states
            )
            plt.title("Unmet Demand Heatmap Over Time by Intensity")
            plt.xlabel("Time")
            plt.ylabel("Intensity")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            save_path = os.path.join("Heat map", "unmet_demand_M1=4_M2=1.png")
            plt.savefig(save_path, dpi=300)
            print(f"Heatmap saved to: {save_path}")

            # Plotting satisfied demand heat map
            plt.figure(figsize=(6, 6))
            sns.heatmap(
                avg_satisfied,               # your masked average satisfied matrix
                cmap="viridis",              
                cbar_kws={'label': 'Average Satisfied Demand'},
                xticklabels=range(1, avg_satisfied.shape[1] + 1),  # time steps: t = 1 to T1 - 1
                yticklabels=range(avg_satisfied.shape[0]),         # intensity: 0 to 5
                linewidths=0.3,
                linecolor='gray',
                square=True,
                mask=np.isnan(avg_satisfied)  # this is key: white-out the unreachable states
            )
            plt.title("Satisfied Demand Heatmap Over Time by Intensity")
            plt.xlabel("Time")
            plt.ylabel("Intensity")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            save_path = os.path.join("Heat map", "satisfied_demand_M1=4_M2=1.png")
            plt.savefig(save_path, dpi=300)
            print(f"Heatmap saved to: {save_path}")
            # =====================================================================

            return [mean_obj, CI, test_time]
        
        elif self.last_stage == 2:
            nbOS = demand_processor.nbOS

            # Load all sample paths from CSV
            OS_paths_df = pd.read_csv(sample_paths_file)
            OS_paths = OS_paths_df.values[:nbOS] # Converts the DataFrame into a NumPy array

            objs = np.zeros((nbOS, self.T1))
            penalty_cost_pre = np.zeros(nbOS)
            penalty_cost_post = np.zeros(nbOS)
            proc_cost_pre = np.zeros(nbOS)
            proc_cost_post = np.zeros(nbOS)
            trans_cost_pre = np.zeros(nbOS)
            trans_cost_post = np.zeros(nbOS)
            unmet_amount = np.zeros(nbOS)
            satisfied_amount = np.zeros(nbOS)
            proc_amount = np.zeros(nbOS)
            trans_amount = np.zeros(nbOS)
            Uval = {}
            fval = {}

            start = time.time()

            # For heatmap
            # Shape: (6 intensity [0–5], T1–1 time steps excluding t=0)
            inventory_sum = np.zeros((6, self.T1 - 1)) # An array holding the sum of I values (per intensity per time) across all sample paths. We don't consider t=0 because we start from one common state.
            unmet_sum = np.zeros((6, self.T1 - 1)) # An array holding the sum of U values (per intensity per time) across all sample paths.
            satisfied_sum = np.zeros((6, self.T1 - 1)) # An array holding the sum of S values (per intensity per time) across all sample paths
            visit_count = np.zeros((6, self.T1 - 1)) # An array to count how many sample paths visit each (intensity, time) pair 
            
            for s in range(nbOS):
                I_prev = np.zeros((self.V, self.R, self.T1))
                U_prev = np.zeros((self.V, self.T1))
                if self.travel_time == 2:
                    f1_prev = np.zeros((self.V, self.V, self.R, self.T1, self.T1))
                for t in range(self.T1):
                # for t in range(self.T1-1):
                    k_t = eval(OS_paths[s, t]) 
                    if t > 0:
                        intensity = k_t[1]  # extract the intensity from (track_error, intensity)
                        visit_count[intensity][t - 1] += 1 # Update count
                        if self.travel_time == 1:
                            self.update_rhs_time1(t, k_t, I_prev, U_prev)
                        elif self.travel_time == 2:
                            self.update_rhs(t, k_t, I_prev, U_prev, f1_prev)                
                    self.m[t, k_t].optimize()
                    if self.m[t, k_t].status != GRB.OPTIMAL:
                        print(" in evaluation")
                        print(f"Model in stage = {t} and state = {k_t}, in forward pass is {self.m[t,k_t].status}")
                        exit(0)
                    else:
                        if t == self.t_L:
                            theta_val = 0
                        else:
                            theta_val = self.theta[t, k_t].x
                        objs[s, t] = self.m[t,k_t].ObjVal - theta_val
                        Uval[s, t] = {}
                        fval[s, t] = {}
                        for i in range(self.V):
                            for r in range(self.R):
                                if t == self.t_L:
                                    I_prev[i, r, t] = self.I[t, k_t][i, r, t].x # Extract inventory level (state var)
                                else:
                                    I_prev[i, r, t] = self.I[t, k_t][i, r].x

                                if t > 0 and i in self.V1: # Only include inventory at demand locations before landfall (shelters) for t > 0
                                    inventory_sum[intensity][t - 1] += I_prev[i, r, t] # For heat map
                            if t == self.t_L:
                                U_prev[i, t] = self.U[t, k_t][i, t].x # Extract demand shortage (state var)
                                unmet_amount[s] += U_prev[i, t] # Accumulate amount of unmet demand at time of landfall
                                satisfied_amount[s] += self.S[t, k_t][i, t].x # Accumulate amount of satisfied demand at time of landfal
                                Uval[s, t][i] = self.U[t, k_t][i, t].x
                                for t_post in range(self.t_L + 1, self.T):
                                    penalty_cost_post[s] += self.p_i[i] * self.U[self.t_L, k_t][i, t_post].x
                                    unmet_amount[s] += self.U[self.t_L, k_t][i, t_post].x # Accumulate amount of unmet demand after landfall
                                    satisfied_amount[s] += self.S[self.t_L, k_t][i, t_post].x # Accumulate amount of satisfied demand after landfal
                            else:
                                U_prev[i, t] = self.U[t, k_t][i].x
                                unmet_amount[s] += U_prev[i, t] # Accumulate amount of unmet demand before landfall
                                satisfied_amount[s] += self.S[t, k_t][i].x # Accumulate amount of satisfied demand before landfall
                                Uval[s, t][i] = self.U[t, k_t][i].x
                            if t > 0 and i in self.V1: # Only include inventory at demand locations before landfall (shelters) for t > 0
                                unmet_sum[intensity][t - 1] += U_prev[i, t] # For heat map
                                if t == self.t_L:
                                    satisfied_sum[intensity][t - 1] += self.S[t, k_t][i, t].x # For heat map
                                else:
                                    satisfied_sum[intensity][t - 1] += self.S[t, k_t][i].x # For heat map
                        
                        if self.travel_time == 2:
                            for i in range(self.V):
                                for j in range(self.V):
                                    if (i, j) in self.A:
                                        for r in range(self.R):
                                            for t_prime in range(t+1):
                                                # if t == self.t_L: ***
                                                #     f1_prev[i, j, r, t_prime, t] = self.f1[t, k_t][i, j, r, t_prime, t].x
                                                # else:
                                                if t < self.t_L:
                                                    f1_prev[i, j, r, t_prime, t] = self.f1[t, k_t][i, j, r, t_prime].x                                      
                        
                        # Computing procurement and transportation cost before landfall
                        for i in range(self.V):
                            for j in range(self.V):
                                if (i, j) in self.A:
                                    for r in range(self.R):
                                        if t < self.t_L:
                                            fval[s, t][i, j, r] = self.f[t, k_t][i, j, r].x
                                        else:
                                            fval[s, t][i, j, r] = self.f[t, k_t][i, j, r, t].x
                                        if i in self.V0:  # Procurement cost only from warehouses
                                            proc_cost_pre[s] += self.cost_ir[i][r] * fval[s, t][i, j, r] 
                                            proc_amount[s] += fval[s, t][i, j, r]                     
                                        trans_cost_pre[s] += self.m_ij[i][j] * fval[s, t][i, j, r]
                                        trans_amount[s] += fval[s, t][i, j, r]
                        
                        
                        # Computing procurement and transportation cost after landfall
                        if t == self.t_L:
                            for t_post in range(self.t_L + 1, self.T):
                                for i in range(self.V):
                                    for j in range(self.V):
                                        if (i, j) in self.A:
                                            for r in range(self.R):
                                                if i in self.V0:  # Procurement only from warehouses
                                                    proc_cost_post[s] += self.cost_ir[i][r] * self.f[self.t_L, k_t][i, j, r, t_post].x
                                                    proc_amount[s] += self.f[self.t_L, k_t][i, j, r, t_post].x
                                                trans_cost_post[s] += self.m_ij[i][j] * self.f[self.t_L, k_t][i, j, r, t_post].x
                                                trans_amount[s] += self.f[self.t_L, k_t][i, j, r, t_post].x



                    # Computing penalty cost before landfall for each sample path
                    for i in self.V1:
                        penalty_cost_pre[s] += self.p_i[i] * Uval[s, t][i]
                    

                # # ========= Plug in the second last stage solution into the last stage problem ===========
                # k_t = eval(OS_paths[s, self.t_L])

                # # # print("Non-zero post-landfall demand values:")
                # # post_landfall_demand = self.demand_processor.msp_post_landfall_demand(k_t, self.network_processor)
                # # for (i, t), val in post_landfall_demand.items():
                # #     if val > 1e-3:
                # #         print(f"demand[{i},{t}] = {val:.2f}")

                # # print("state:", k_t)
                # # Realized demand at stage t, given Markov state k_t
                # demand_real = self.demand_processor.msp_demand_mapping(self.t_L, k_t, self.network_processor)
                # # print("demand[6]:", demand_real[6])
                # # print("demand[7]:", demand_real[7])
                # # print("demand[8]:", demand_real[8])
                # # print("demand[9]:", demand_real[9])
                # # print("demand[10]:", demand_real[10])
                # # Demand shortage rhs
                # for i in range(self.V):  # Loop over all locations/nodes
                #     self.demand_shortage_constr1[self.t_L, k_t][i].setAttr(GRB.Attr.RHS, U_prev[i, self.t_L - 1] + demand_real[i])
                #     self.demand_shortage_constr2[self.t_L, k_t][i].setAttr(GRB.Attr.RHS, U_prev[i, self.t_L - 1] + demand_real[i])

                # # Inventory flow-balance rhs
                # # We are at stage t and updating rhs for constraints in stage t, so we check the condition at this stage but add the state variables from previous stage.
                # for i in range(self.V):
                #     for r in range(self.R):
                #         if self.travel_time == 1:
                #             rhs_val = I_prev[i, r, self.t_L - 1]
                #         elif self.travel_time == 2:
                #             rhs_val = I_prev[i, r, self.t_L - 1] + sum(f1_prev[j, i, r, t_prime, self.t_L - 1] for j in range(self.V) for t_prime in range(self.t_L) if (j, i) in self.A and (j, i, t_prime, self.t_L) in self.delta)
                #         self.inventory_update_constr1[self.t_L, k_t][i, r].setAttr(GRB.Attr.RHS, rhs_val)
                #         self.inventory_update_constr2[self.t_L, k_t][i, r].setAttr(GRB.Attr.RHS, rhs_val)

                # # In-transit flow rhs
                # if self.travel_time == 2:
                #     for i in range(self.V):
                #         for j in range(self.V):
                #             if (i, j) in self.A:
                #                 for r in range(self.R):
                #                     for t_prime in range(self.t_L):  # Only for past departure times
                #                         if (i, j, t_prime, self.t_L) not in self.delta: # Check whether flow (i, j) started at t′ is still in transit at stage t
                #                             rhs_val = f1_prev[i, j, r, t_prime, self.t_L - 1] # If it hasn't arrived yet, carry forward from previous stage
                #                         else:
                #                             rhs_val = 0.0 # If it has arrived, flow is no longer in transit
                #                         self.in_transit_update_constr1[self.t_L, k_t][i, j, r, t_prime].setAttr(GRB.Attr.RHS, rhs_val)
                #                         self.in_transit_update_constr2[self.t_L, k_t][i, j, r, t_prime].setAttr(GRB.Attr.RHS, rhs_val)

                # # self.m[self.t_L, k_t].write(f"LP/last_stage_model_t{self.t_L}_state{k_t}.lp")  # Save LP file
                # self.m[self.t_L, k_t].optimize()
                # if self.m[self.t_L, k_t].status != GRB.OPTIMAL:
                #     print(" in evaluation")
                #     print(f"Model in stage = {self.t_L} and state = {k_t}, in forward pass is {self.m[self.t_L,k_t].status}")
                #     exit(0)
                # else:
                #     print("Solution of the last stage problem:")
                #     # Print Local Flow (f)
                #     for i in range(self.V):
                #         for j in range(self.V):
                #             if (i, j) in self.A:
                #                 for r in range(self.R):
                #                     for t_post in range(self.t_L, self.T):
                #                         val = self.f[self.t_L, k_t][i, j, r, t_post].x
                #                         if val > 1e-6:
                #                             print(f"f[{i},{j},{r},{t_post}] = {val}")

                #     # Print Inventory (I)
                #     for i in range(self.V):
                #         for r in range(self.R):
                #             for t_post in range(self.t_L, self.T):
                #                 val = self.I[self.t_L, k_t][i, r, t_post].x
                #                 if val > 1e-6:
                #                     print(f"I[{i},{r},{t_post}] = {val}")

                #     # Print Unmet Demand (U)
                #     for i in range(self.V):
                #         for t_post in range(self.t_L, self.T):
                #                 val = self.U[self.t_L, k_t][i, t_post].x
                #                 if val > 1e-6:
                #                     print(f"U[{i},{t_post}] = {val}")

                #     # Print Satisfied Demand (S)
                #     for i in range(self.V):
                #         for t_post in range(self.t_L, self.T):
                #             val = self.S[self.t_L, k_t][i, t_post].x
                #             if val > 1e-6:
                #                 print(f"S[{i},{t_post}] = {val}")

                #     # Print In-Transit Flows (f1)
                #     if self.travel_time == 2:
                #         for i in range(self.V):
                #             for j in range(self.V):
                #                 if (i, j) in self.A:
                #                     for r in range(self.R):
                #                         for t_prime in range(self.t_L+1):
                #                             for t_post in range(self.t_L, self.T):
                #                                 val = self.f1[self.t_L, k_t][i, j, r, t_prime, t_post].x
                #                                 if val > 1e-6:
                #                                     print(f"f1[{i},{j},{r},{t_prime},{t_post}] = {val}")
                # ======================================================================================================



            # Printing amount of variables
            print("Average amount of unmet demand=", np.mean(unmet_amount))
            print("Average amount of satisfied demand=", np.mean(satisfied_amount))
            print("Average amount of procurement=", np.mean(proc_amount))
            print("Average amount of transportation=", np.mean(trans_amount))
            
            # Printing costs
            print("Average Penalty Cost Pre =", np.mean(penalty_cost_pre))
            print("Average Penalty Cost Post =", np.mean(penalty_cost_post))
            print("Average Procurement Cost Pre =", np.mean(proc_cost_pre))
            print("Average Procurement Cost Post =", np.mean(proc_cost_post))
            print("Average Transportation Cost Pre =", np.mean(trans_cost_pre))
            print("Average Transportation Cost Post =", np.mean(trans_cost_post))


            mean_obj = np.mean(np.sum(objs, axis=1))
            std_obj = np.std(np.sum(objs, axis=1))
            ci_low = mean_obj - 1.96 * std_obj / np.sqrt(nbOS)
            ci_high = mean_obj + 1.96 * std_obj / np.sqrt(nbOS)
            CI = mean_obj - ci_low
            print(f"\nSDDP Evaluation Results")
            print(f"μ ± 1.96*σ/√NS = {mean_obj} ± {CI}")
            print(f"95% Confidence Interval = [{ci_low:.2f}, {ci_high:.2f}]")
            test_time = time.time() - start
            
            # for t in range(self.T1 - 1):  # time steps 1 to T1-1
            #     print(f"\nTime step t = {t + 1}:")
            #     for i in range(6):  # intensity levels 0 to 5
            #         print(f"  Intensity {i}: {int(visit_count[i][t])} paths")

            # ========================= Heat map ==============================
            # Computing average inventory per intensity per stage
            avg_inventory = np.divide(inventory_sum, visit_count, out=np.full_like(inventory_sum, np.nan), where=visit_count != 0)
            # print("Inventory:\n", avg_inventory)
            avg_unmet = np.divide(unmet_sum, visit_count, out=np.full_like(unmet_sum, np.nan), where=visit_count != 0)
            # print("Unmet:\n", avg_unmet)
            avg_satisfied = np.divide(satisfied_sum, visit_count, out=np.full_like(satisfied_sum, np.nan), where=visit_count != 0)
            # print("Satisfied:\n", avg_satisfied)
            
            # Plotting inventory heat map
            plt.figure(figsize=(6, 6))
            sns.heatmap(
                avg_inventory,               # your masked average inventory matrix
                cmap="viridis",              
                cbar_kws={'label': 'Average Inventory'},
                xticklabels=range(1, avg_inventory.shape[1] + 1),  # time steps: t = 1 to T1 - 1
                yticklabels=range(avg_inventory.shape[0]),         # intensity: 0 to 5
                linewidths=0.3,
                linecolor='gray',
                square=True,
                mask=np.isnan(avg_inventory)  # this is key: white-out the unreachable states
            )
            plt.title("Inventory Heatmap Over Time by Intensity")
            plt.xlabel("Time")
            plt.ylabel("Intensity")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            save_path = os.path.join("Heat map", "inventory_M1=1_M2=1.png")
            plt.savefig(save_path, dpi=300)
            print(f"Heatmap saved to: {save_path}")
            
            # Plotting unmet demand heat map
            plt.figure(figsize=(6, 6))
            sns.heatmap(
                avg_unmet,               # your masked average unmet matrix
                cmap="viridis",              
                cbar_kws={'label': 'Average Unmet Demand'},
                xticklabels=range(1, avg_unmet.shape[1] + 1),  # time steps: t = 1 to T1 - 1
                yticklabels=range(avg_unmet.shape[0]),         # intensity: 0 to 5
                linewidths=0.3,
                linecolor='gray',
                square=True,
                mask=np.isnan(avg_unmet)  # this is key: white-out the unreachable states
            )
            plt.title("Unmet Demand Heatmap Over Time by Intensity")
            plt.xlabel("Time")
            plt.ylabel("Intensity")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            save_path = os.path.join("Heat map", "unmet_demand_M1=1_M2=1.png")
            plt.savefig(save_path, dpi=300)
            print(f"Heatmap saved to: {save_path}")

            # Plotting satisfied demand heat map
            plt.figure(figsize=(6, 6))
            sns.heatmap(
                avg_satisfied,               # your masked average satisfied matrix
                cmap="viridis",              
                cbar_kws={'label': 'Average Satisfied Demand'},
                xticklabels=range(1, avg_satisfied.shape[1] + 1),  # time steps: t = 1 to T1 - 1
                yticklabels=range(avg_satisfied.shape[0]),         # intensity: 0 to 5
                linewidths=0.3,
                linecolor='gray',
                square=True,
                mask=np.isnan(avg_satisfied)  # this is key: white-out the unreachable states
            )
            plt.title("Satisfied Demand Heatmap Over Time by Intensity")
            plt.xlabel("Time")
            plt.ylabel("Intensity")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            save_path = os.path.join("Heat map", "satisfied_demand_M1=1_M2=1.png")
            plt.savefig(save_path, dpi=300)
            print(f"Heatmap saved to: {save_path}")
            # =====================================================================


            return [mean_obj, CI, test_time]



    
        
 




            


