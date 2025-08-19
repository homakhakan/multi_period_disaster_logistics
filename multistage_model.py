import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import msp_helper
import time
from openpyxl import load_workbook
import os

class MultiStageSPModel:
    def __init__(self, network_processor, demand_processor, params, travel_time):
        self.network_processor = network_processor
        self.demand_processor = demand_processor
        self.params = params
        self.travel_time = travel_time
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
        self.T_Bef = network_processor.T_Bef
        self.t_ijt = network_processor.t_ijt
        self.t_L = network_processor.t_L 

        # Define delta indicator
        if self.travel_time == 2:
            self.delta = []
            for (i, j) in self.A:
                for t in range(len(self.T_Bef)):
                    t_prime = t - self.t_ijt[i][j] - 1
                    if 0 <= t_prime:             
                        self.delta.append((i, j, t, t_prime))

            # Convert to set for fast membership checking
            self.delta = set(self.delta)

        # Markov chain data
        transition_path = "Data/pi_mssp_d_ST_track_15.json"
        self.transition_dict = msp_helper.load_transition_dict(transition_path)
        self.nodeLists = msp_helper.build_node_lists(self.transition_dict)

        self.timer_stats = {
            "rhs_inventory_loop": 0.0,
            "rhs_inventory_get_fprev": 0.0,
            "rhs_inventory_setattr": 0.0,
            "rhs_intransit_loop": 0.0,
            "rhs_intransit_get_fprev": 0.0,
            "rhs_intransit_setattr": 0.0,
            "rhs_total": 0.0
        }

    def stage_t_state_k_problem(self, t, k, last_stage=False):
        """
        Build the optimization model for *one* node (t, k) in the scenario tree.
        t : Stage index (time period).
        k : Markov-chain state realised
        last_stage : If True, we are at the deterministic multi-period terminal stage (θ removed).
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
                            for t_prime in range(0, t):
                                f1[i, j, r, t_prime] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f1_{i}_{j}_{r}_{t_prime}")

                            # NEW departure time  t′ = t   (needed for last constraint)
                            f1[i, j, r, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f1_{i}_{j}_{r}_{t}")

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

        # We need to define the last-stage problem separately since we do not have the theta variable there
        if not last_stage:
            theta = m.addVar(lb=0, name="theta")
        else:
            theta = None

        # Constraints
        # Demand shortage update
        demand_shortage_constr = {}
        for i in range(self.V):
            demand_shortage_constr[i] = m.addConstr(U[i] + S[i] == 0, name=f"DemandShortageUpdate_{i}")

        # Inventory update
        if self.travel_time == 1:
            inventory_update_constr = {}
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
                        inventory_update_constr[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) - gp.quicksum(f[j, i, r] for j in range(self.V) if (j, i) in self.A)
                                                                    + S[i] * g_coeff == self.network_processor.I_ir0[i][r], name=f"InventoryUpdate_{i}_{r}_t0")
                    else:
                        inventory_update_constr[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) - gp.quicksum(f[j, i, r] for j in range(self.V) if (j, i) in self.A)
                                                                    + S[i] * g_coeff == 0, name=f"InventoryUpdate_{i}_{r}_t{t}")         
        
        elif self.travel_time == 2:
            inventory_update_constr = {}
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
                        inventory_update_constr[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) 
                                                                    + S[i] * g_coeff == self.network_processor.I_ir0[i][r], name=f"InventoryUpdate_{i}_{r}_t0")
                    else:
                        inventory_update_constr[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) 
                                                                    + S[i] * g_coeff == 0, name=f"InventoryUpdate_{i}_{r}_t{t}") 

        # Flow capacity
        flow_capacity_constr = {}
        for i in range(self.V):
            for j in range(self.V):
                if (i, j) in self.A:
                    cap_ij_t = self.K_ijt[i][j][t]
                    # # Debugging
                    # if cap_ij_t > 1e-6:  # Only print nonzero capacities
                    #     print(f"Cap[{i},{j},{t}] = {cap_ij_t}")
                    # print("w_r     =", self.w_r)
                    flow_capacity_constr[i, j] = m.addConstr(gp.quicksum(self.w_r[r] * f[i, j, r] for r in range(self.R)) <= cap_ij_t, name=f"FlowCapacity_{i}_{j}_{t}")

        if self.travel_time == 2:
            # In-transit flow update to the next stage
            in_transit_update_constr = {}
            for i in range(self.V):
                for j in range(self.V):
                    if (i, j) in self.A:
                        for r in range(self.R):
                            # t′ < t: Carry forward past flows (to be updated later)
                            for t_prime in range(0, t):
                                in_transit_update_constr[i, j, r, t_prime] = m.addConstr(f1[i, j, r, t_prime] == 0, name=f"CarryTransit_{i}_{j}_{r}_{t_prime}_to_{t}")

                            # t′ = t: Assign new flow to f1 for the next stage
                            in_transit_update_constr[i, j, r, t] = m.addConstr(f1[i, j, r, t] == f[i, j, r], name=f"InTransitNew_{i}_{j}_{r}_{t}")

        # Objective
        objective = (
            gp.quicksum(self.p_i[i] * U[i] for i in range(self.V)) 
            + gp.quicksum(
                gp.quicksum(self.cost_ir[i][r] * f[i, j, r] for i in self.V0 for j in range(self.V) if (i, j) in self.A)
                + gp.quicksum(self.m_ij[i][j] * f[i, j, r] for i in range(self.V) for j in range(self.V) if (i, j) in self.A) for r in range(self.R))
            + (theta if theta is not None else 0)
        )
        

        m.setObjective(objective, GRB.MINIMIZE)

        if self.travel_time == 1:
            return {
            "model": m,
            "I": I,                    # state variable: inventory
            "U": U,                    # state variable: unmet demand
            "f": f,                    # local variable: flow
            "S": S,                    # local variable: satisfied demand
            "theta": theta,            # cost-to-go (None in last stage)
            "demand_shortage_constr": demand_shortage_constr,
            "inventory_update_constr": inventory_update_constr,
            "flow_capacity_constr": flow_capacity_constr,
            }
        elif self.travel_time == 2:
            return {
            "model": m,
            "I": I,                    # state variable: inventory
            "U": U,                    # state variable: unmet demand
            "f1": f1,                  # state variable: in-transit flow
            "f": f,                    # local variable: flow
            "S": S,                    # local variable: satisfied demand
            "theta": theta,            # cost-to-go (None in last stage)
            "demand_shortage_constr": demand_shortage_constr,
            "inventory_update_constr": inventory_update_constr,
            "flow_capacity_constr": flow_capacity_constr,
            "in_transit_update_constr": in_transit_update_constr
            }
    
    def build_models(self):
        """ Define optimization models for each (t, k) using the transition JSON file and msp_helper tools."""

        # Prepare dictionaries
        self.models = {}
        self.I = {}
        self.U = {}        
        self.f = {}
        self.S = {}
        self.theta = {}
        self.demand_shortage_constr = {}
        self.inventory_update_constr = {}
        self.flow_capacity_constr = {}

        if self.travel_time == 2:
            self.f1 = {}
            self.in_transit_update_constr = {}

        # Loop over pre-landfall stochastic stages (t = 1 … t_L)
        for t in range(len(self.T_Bef)):
            for k in self.nodeLists[t]: # Every reachable Markov state at time t
                last_stage = (t == self.t_L) # True only at t = t_L

                # Build the stage-state model and unpack everything we need
                model_dict = self.stage_t_state_k_problem(t, k, last_stage=last_stage)

                self.models[t, k] = model_dict["model"]
                self.I[t, k] = model_dict["I"]
                self.U[t, k] = model_dict["U"]   
                self.f[t, k] = model_dict["f"]
                self.S[t, k] = model_dict["S"]
                self.theta[t, k] = model_dict["theta"]
                self.demand_shortage_constr[t, k] = model_dict["demand_shortage_constr"]
                self.inventory_update_constr[t, k] = model_dict["inventory_update_constr"]
                self.flow_capacity_constr[t, k] = model_dict["flow_capacity_constr"]
            
                if self.travel_time == 2:
                    self.f1[t, k] = model_dict["f1"]
                    self.in_transit_update_constr[t, k] = model_dict["in_transit_update_constr"]

    def forward_pass(self):

        k_t = self.nodeLists[0][0]      # First state at t=0
        state_path = [k_t]              # Keep track of visited states
        thetaval = [0] * len(self.T_Bef)     # Stores theta values (cost-to-go)

        # Initialize the state variables from t=0
        I_prev = {(i, r): 0 for i in range(self.V) for r in range(self.R)}
        U_prev = {i: 0 for i in range(self.V)}
        if self.travel_time == 2:
            f_prev = {(i, j, r, t_prime): 0 for (i, j) in self.A for r in range(self.R) for t_prime in range(len(self.T_Bef))} # No past flows exist before t=1

        lb = self.lower_bound

        for t in range(len(self.T_Bef)):
            if t > 0:
                k_t = msp_helper.MC_sample(t - 1, k_t, self.transition_dict)
                state_path.append(k_t)          # Keep sample path history
                if self.travel_time == 1:
                    self.mspFP_update_rhs_time1(t, k_t, I_prev, U_prev)
                elif self.travel_time == 2:
                    self.mspFP_update_rhs(t, k_t, I_prev, U_prev, f_prev) # Update the right hand side of the constraints
            model = self.models[t, k_t]

            # if t == 8:
            #     print(f"Writing LP for t={t}, k={k_t}")
            #     model.write(f"debugging/model_t{t}_k{k_t}.lp")

            model.optimize() # Optimize the model

            # print(f"\n--- Debug: Decision Variables at t={t}, k_t={k_t} ---")
            # print(f"Objective Value: {model.ObjVal:.2f}")

            # f: shipment flow
            # print("\nNon-zero f[i,j,r]:")
            for (i, j, r), var in self.f[t, k_t].items():
                if var.X > 1e-6:
                    print(f"  f[{i},{j},{r}] = {var.X:.2f}")

            # # I: inventory
            # print("\nNon-zero I[i,r]:")
            # for (i, r), var in self.I[t, k_t].items():
            #     if var.X > 1e-6:
            #         print(f"  I[{i},{r}] = {var.X:.2f}")

            # # U: unmet demand
            # print("\nNon-zero U[i]:")
            # for i, var in self.U[t, k_t].items():
            #     if var.X > 1e-6:
            #         print(f"  U[{i}] = {var.X:.2f}")

            # S: satisfied demand
            # print("\nNon-zero S[i]:")
            for i, var in self.S[t, k_t].items():
                if var.X > 1e-6:
                    print(f"  S[{i}] = {var.X:.2f}")

            # # Theta (if exists)
            # theta_var = self.theta[t, k_t]
            # if theta_var is not None and theta_var.X > 1e-6:
            #     print(f"\nTheta = {theta_var.X:.2f}")

            # # f1: in-transit flow
            # print("\nNon-zero f1[i,j,r,t']:")        
            # for (i, j, r, t_prime), var in self.f1[t, k_t].items():
            #     if var.X > 1e-6:
            #         print(f"  f1[{i},{j},{r},{t_prime}] = {var.X:.2f}")

            # Check for the optimality and then extract the solution for updating state variables
            if model.status != gp.GRB.OPTIMAL:
                raise RuntimeError(f"Model at t={t}, state={k_t} not optimal")
            else:
                I_prev = {(i, r): self.I[t, k_t][i, r].X for i in range(self.V) for r in range(self.R)} # Extract inventory level (state var)
                # print(f"Stage {t} - Nonzero Inventory Levels:") # Debugging
                # for (i, r), value in I_prev.items():
                #     if value > 1e-6:
                #         print(f"  Location {i}, Resource {r} → Inventory: {value:.2f}")

                U_prev = {i: self.U[t, k_t][i].X for i in range(self.V)} # Extract demand shortage (state var)

                if self.travel_time == 2:
                    # Extract the new in-transit flow for next stage
                    f_current = {(i, j, r, t_prime): 0 for (i, j) in self.A for r in range(self.R) for t_prime in range(len(self.T_Bef))}
                    # Carry forward earlier shipments t' < t that have NOT arrived
                    for (i, j) in self.A:
                        for r in range(self.R):
                            for t_prime in range(0, t):              # Only past departures
                                if (i, j, t, t_prime) not in self.delta:
                                    f_current[(i, j, r, t_prime)] = f_prev[(i, j, r, t_prime)] # still travelling, keep it

                    # Add NEW shipments that depart at the current stage t
                    for (i, j, r), var in self.f[t, k_t].items():  # local decision variable
                        if var.X > 1e-6:
                            print("f variable=", var.X) # Debugging
                            f_current[(i, j, r, t)] = var.X    
                    # Save it to use as f_prev when we move to stage t+1
                    f_prev = f_current

                thetaval[t] = self.theta[t, k_t].x if self.theta[t, k_t] is not None else 0.0 # To avoid AttributeError: 'NoneType' object has no attribute 'x'

                # Debugging
                #print(f"Forward pass | t={t} | visited state = {k_t} | theta = {thetaval[t]:.2f}")

                if t == 0:
                    lb = model.objVal               
        # print("state_path:", state_path)
        if self.travel_time == 1:
            return I_prev, U_prev, thetaval, lb, state_path
        elif self.travel_time == 2:
            return I_prev, U_prev, f_prev, thetaval, lb, state_path

    def mspFP_update_rhs(self, t, k_t, I_prev, U_prev, f_prev):
        """Update right-hand sides of the (t , k_t) model constraints before optimization.""" 
        
        start_total = time.time()

        # Realized demand at stage t, given Markov state k_t
        demand_real = self.demand_processor.msp_demand_mapping(t, k_t, self.network_processor)
    
        # Demand shortage rhs
        for i, cons in self.demand_shortage_constr[t, k_t].items():
            rhs = U_prev[i] + demand_real[i]
            cons.setAttr(GRB.Attr.RHS, rhs)

        # # Debugging
        # debug_data = []
        # for i in self.demand_shortage_constr[t, k_t].keys():
        #     debug_data.append({"Location i": i, "U_prev[i]": U_prev[i], "demand_real[i]": demand_real[i], "RHS": U_prev[i] + demand_real[i]})

        # # Create debugging folder if it doesn't exist
        # if not os.path.exists("debugging"):
        #     os.makedirs("debugging")

        # # Save to Excel
        # df_debug = pd.DataFrame(debug_data)
        # df_debug.to_excel(f"debugging/demand_shortage_rhs_t{t}_k{k_t}.xlsx", index=False)

        inv_get_fprev_time = 0
        inv_setattr_time   = 0
        start_inv_loop = time.time()

        # Inventory flow-balance rhs
        for (i, r), cons in self.inventory_update_constr[t, k_t].items():
            arrivals = 0
            for (j, i2, t2, t_prime) in self.delta:
                if t2 == t and i2 == i:  # only relevant (j → i, t, t′)
            # for j in range(self.V):
            #     if (j, i) in self.A:
            #         for t_prime in range(0, t):
            #             if (j, i, t, t_prime) in self.delta:
                            start_get = time.time()
                            arrivals += f_prev[(j, i, r, t_prime)]
                            # if arrivals > 1e-6: # Debugging
                            #     print("arrivals=", arrivals)
                            inv_get_fprev_time += time.time() - start_get
            rhs = I_prev[(i, r)] + arrivals
            start_set = time.time()
            cons.setAttr(GRB.Attr.RHS, rhs)
            inv_setattr_time += time.time() - start_set
        
        self.timer_stats["rhs_inventory_loop"] += time.time() - start_inv_loop
        self.timer_stats["rhs_inventory_get_fprev"] += inv_get_fprev_time
        self.timer_stats["rhs_inventory_setattr"] += inv_setattr_time
        
        intransit_get_fprev_time = 0
        intransit_setattr_time   = 0
        start_intransit_loop = time.time()

        # In-transit flow rhs
        for (i, j, r, t_prime), cons in self.in_transit_update_constr[t, k_t].items():
            if (i, j, t, t_prime) not in self.delta:
                start_get = time.time()
                rhs = f_prev[(i, j, r, t_prime)]
                # if rhs > 1e-6: # Debugging
                #     print("rhs=", rhs)
                intransit_get_fprev_time += time.time() - start_get
            else:
                rhs = 0.0

            start_set = time.time()
            cons.setAttr(GRB.Attr.RHS, rhs)
            intransit_setattr_time += time.time() - start_set

        self.timer_stats["rhs_intransit_loop"] += time.time() - start_intransit_loop
        self.timer_stats["rhs_intransit_get_fprev"] += intransit_get_fprev_time
        self.timer_stats["rhs_intransit_setattr"] += intransit_setattr_time

        self.timer_stats["rhs_total"] += time.time() - start_total

    def mspFP_update_rhs_time1(self, t, k_t, I_prev, U_prev):
        """Update right-hand sides of the (t , k_t) model constraints before optimization.""" 
        
        # Realized demand at stage t, given Markov state k_t
        demand_real = self.demand_processor.msp_demand_mapping(t, k_t, self.network_processor)
        
        # print(f"demand= {k_t}, {t}, {demand_real}")
        # print("U_prev=", U_prev)
        # Demand shortage rhs
        for i, cons in self.demand_shortage_constr[t, k_t].items():
            rhs = U_prev[i] + demand_real[i]
            cons.setAttr(GRB.Attr.RHS, rhs)

        # Inventory flow-balance rhs
        for (i, r), cons in self.inventory_update_constr[t, k_t].items():
            rhs = I_prev[(i, r)]
            cons.setAttr(GRB.Attr.RHS, rhs)

    def backward_pass(self, I_prev, U_prev, f_prev, thetaval, state_path):
        
        cutviolFlag = False

        for t in range(len(state_path)-1, 0, -1):
            Q = {}  # Use dictionary to store: key = Markov state (tuple), value = objVal
            # List for all the dual multipliers of the constraints
            dual_demand    = {}  # (k,i)              
            dual_inventory = {}  # (k,i,r)            
            dual_capacity  = {}  # (k,i,j)            
            dual_transit   = {}  # (k,i,j,r,t′)    

            sample_state = state_path[t - 1]  # The state observed at time t-1

            for k_state in self.nodeLists[t]:
                self.mspFP_update_rhs(t, k_state, I_prev, U_prev, f_prev)
                model = self.models[t, k_state]
                model.optimize()

                if model.status != gp.GRB.OPTIMAL:
                    raise RuntimeError(f"Model at t={t}, state={k_state} not optimal in backward pass.")
                    
                else:
                    Q[k_state] = model.ObjVal    # Save the objective value for this (t , k_state)

                    # Extract dual values for each constraint set
                    for i in range(self.V):
                        dual_demand[k_state, i] = self.demand_shortage_constr[t, k_state][i].pi

                    for i in range(self.V):
                        for r in range(self.R):
                            dual_inventory[k_state, i, r] = self.inventory_update_constr[t, k_state][i, r].pi

                    for i in range(self.V):
                        for j in range(self.V):
                            if (i, j) in self.A:
                                dual_capacity[k_state, i, j] = self.flow_capacity_constr[t, k_state][i, j].pi

                    for i in range(self.V):
                        for j in range(self.V):
                            if (i, j) in self.A:
                                for r in range(self.R):
                                    for t_prime in range(0, t):
                                        dual_transit[k_state, i, j, r, t_prime] = self.in_transit_update_constr[t, k_state][i, j, r, t_prime].pi

            # Solving all stage-(t-1) problems and generate cuts/valid inequalities
            for n in self.nodeLists[t - 1]:
                Qvalue = 0
                for k_state in self.nodeLists[t]:
                    # Accumulate expected cost-to-go from all reachable next-stage states with non-negligible transition probability
                    if self.transition_dict[str(t-1)][str(n)][str(k_state)] > self.smallest_trans_prob:
                        Qvalue += Q[k_state] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                # Debugging
                #print(f"t={t}, n={n}, Qvalue={Qvalue:.2f}, theta={thetaval[t-1]:.2f}, Δ={(Qvalue - thetaval[t - 1]):.2f}")

                # Check if cut is violated at the sample path encountered in the forward pass (Only interested in checking cut violation on the same node as the forward pass)
                # n  : current state at stage t-1
                # sample_state : state that was actually visited in the forward pass
                if n == sample_state and (
                        (Qvalue - thetaval[t - 1]) / max(1e-10, abs(thetaval[t - 1])) > self.cutviol 
                        and abs(Qvalue - thetaval[t - 1]) > self.cutviol
                ):
                    cutviolFlag = True
                    #print("here")

                # Initialize coeff-dictionaries for each state variable
                cutcoef_I = {(i, r): 0.0 for i in range(self.V) for r in range(self.R)}
                cutcoef_U = {i: 0.0 for i in range(self.V)}
                cutcoef_f1 = {}               
                cut_rhs = 0.0   

                for k_state in self.nodeLists[t]:
                    if self.transition_dict[str(t-1)][str(n)][str(k_state)] > self.smallest_trans_prob:

                        # Demand-shortage constraints
                        for i in range(self.V):
                            lam = dual_demand[k_state, i] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                            cutcoef_U[i] += lam                                                 # Coefficient
                            cut_rhs += lam * U_prev[i]                                          # RHS

                        # Inventory-balance constraints
                        for i in range(self.V):
                            for r in range(self.R):
                                lam = dual_inventory[k_state, i, r] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                cutcoef_I[i, r] += lam                                          # Coefficient
                                cut_rhs += lam * I_prev[i, r]                                   # RHS

                        # Flow capacity constraints
                        for i in range(self.V):
                            for j in range(self.V):
                                if (i, j) in self.A:
                                    lam = dual_capacity[k_state, i, j] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                    # Coefficient multiplies fixed parameter K_ijt, so only RHS:
                                    cut_rhs += lam * self.K_ijt[i][j][t-1]

                        # In-transit flow constraints
                        for i in range(self.V):
                            for j in range(self.V):
                                if (i, j) in self.A:
                                    for r in range(self.R):
                                        for t_prime in range(0, t):
                                            key = (k_state, i, j, r, t_prime)
                                            if key in dual_transit:
                                                lam = dual_transit[key] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                                cutcoef_f1[i, j, r, t_prime] = cutcoef_f1.get((i, j, r, t_prime), 0.0) + lam
                                                #cutcoef_f1[i, j, r, t_prime] = cutcoef_f1[(i, j, r, t_prime)] + lam
                                                cut_rhs += lam * f_prev[(i, j, r, t_prime)] ### check .get
                                                #cut_rhs += lam * f_prev[i, j, r, t_prime]

                # Build the left-hand-side linear expression
                lhs_expr  = self.theta[t-1, n]
                lhs_expr += -gp.quicksum(cutcoef_U[i] * self.U[t-1, n][i] for i in range(self.V))
                lhs_expr += -gp.quicksum(cutcoef_I[i, r] * self.I[t-1, n][i, r] for i in range(self.V) for r in range(self.R))
                lhs_expr += -gp.quicksum(coeff * self.f1[t-1, n][key] for key, coeff in cutcoef_f1.items())
                # if t == self.t_L:
                #     print("cutcoef_U = ")
                #     for i in range(self.V):
                #         if abs(cutcoef_U[i]) > 1e-6:
                #             print(f"{i}, {cutcoef_U[i]}")
                #     print("cutcoef_I = ")
                #     for i in range(self.V):
                #         for r in range(self.R):
                #             if abs(cutcoef_I[i, r]) > 1e-6:
                #                 print(f"{i}, {cutcoef_I[i,r]}")
                #     print("cutcoef_f1 = ")
                #     for key, coeff in cutcoef_f1.items():
                #         if abs(coeff) > 1e-6:
                #             print(f"{key},{coeff}")
                #     #exit(0)

                # Add the cut
                self.models[t-1, n].addConstr(lhs_expr >= Qvalue - cut_rhs, name=f"OptCut_{t-1}_{n}_{iter}")

        return cutviolFlag

    def backward_pass_time1(self, I_prev, U_prev, thetaval, state_path):
                
        cutviolFlag = False

        for t in range(len(state_path)-1, 0, -1):
            #print("t=", t)
            Q = {}  # Use dictionary to store: key = Markov state (tuple), value = objVal
            # List for all the dual multipliers of the constraints
            dual_demand    = {}  # (k,i)              
            dual_inventory = {}  # (k,i,r)            
            dual_capacity  = {}  # (k,i,j)              

            sample_state = state_path[t - 1]  # The state observed at time t-1
            # print("sample_state:", sample_state)
            # print("list of nodes t:", self.nodeLists[t]) # List of all possible states at stage t
            for k_state in self.nodeLists[t]: 
                self.mspFP_update_rhs_time1(t, k_state, I_prev, U_prev)
                model = self.models[t, k_state]
                model.optimize()

                if model.status != gp.GRB.OPTIMAL:
                    raise RuntimeError(f"Model at t={t}, state={k_state} not optimal in backward pass.")
                    
                else:
                    Q[k_state] = model.ObjVal    # Save the objective value for this (t , k_state)
                    # print("Q=", Q[k_state])
                    # Extract dual values for each constraint set
                    # print("========t:", t)
                    # print("dual demand:")
                    for i in range(self.V):
                        dual_demand[k_state, i] = self.demand_shortage_constr[t, k_state][i].pi
                        if dual_demand[k_state, i] < -1e-6:
                            print(f"{k_state}, {i}, {dual_demand[k_state, i]}")

                    # print("========t:", t)
                    # print("dual inventory:")
                    for i in range(self.V):
                        for r in range(self.R):
                            dual_inventory[k_state, i, r] = self.inventory_update_constr[t, k_state][i, r].pi
                            # if dual_inventory[k_state, i, r] > 1e-6:
                            #     print(f"{k_state}, {i}, {r} {dual_inventory[k_state, i, r]}")
                    # print("dual capacity:")
                    for i in range(self.V):
                        for j in range(self.V):
                            if (i, j) in self.A:
                                dual_capacity[k_state, i, j] = self.flow_capacity_constr[t, k_state][i, j].pi
                                # if abs(dual_capacity[k_state, i, j]) > 1e-6:
                                #     print(f"({k_state}, {i}, {j})= {dual_capacity[k_state, i, j]}")
                # exit(0)
                # # Print only for the current sample path node
                # if k_state == state_path[t]:
                #     print(f"\n[Stage {t}] Dual demand for node {k_state} (sample path):")
                #     for i in range(self.V):
                #         for r in range(self.R):
                #             val = dual_inventory[k_state, i, r]
                #             if abs(val) > 1e-6:
                #                 print(f"  Location {i}, Resource {r}: Dual = {val:.4f}")

            # Solving all stage-(t-1) problems and generate cuts/valid inequalities
            for n in self.nodeLists[t - 1]:
                # print("t-1=", t - 1)
                # print("n:", n)
                # print("list of nodes t-1:", self.nodeLists[t - 1])
                # exit(0)
                Qvalue = 0
                for k_state in self.nodeLists[t]:
                    # Accumulate expected cost-to-go from all reachable next-stage states with non-negligible transition probability
                    if self.transition_dict[str(t-1)][str(n)][str(k_state)] > self.smallest_trans_prob:
                        Qvalue += Q[k_state] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                # Debugging
                #print(f"t={t}, n={n}, Qvalue={Qvalue:.2f}, theta={thetaval[t-1]:.2f}, Δ={(Qvalue - thetaval[t - 1]):.2f}")

                # Check if cut is violated at the sample path encountered in the forward pass (Only interested in checking cut violation on the same node as the forward pass)
                # n  : current state at stage t-1
                # sample_state : state that was actually visited in the forward pass
                if n == sample_state and (
                        (Qvalue - thetaval[t - 1]) / max(1e-10, abs(thetaval[t - 1])) > self.cutviol 
                        and abs(Qvalue - thetaval[t - 1]) > self.cutviol
                ):
                    cutviolFlag = True

                # Initialize coeff-dictionaries for each state variable
                cutcoef_I = {(i, r): 0.0 for i in range(self.V) for r in range(self.R)}
                cutcoef_U = {i: 0.0 for i in range(self.V)}             
                cut_rhs = 0.0   

                for k_state in self.nodeLists[t]:
                    if self.transition_dict[str(t-1)][str(n)][str(k_state)] > self.smallest_trans_prob:

                        # Demand-shortage constraints
                        for i in range(self.V):
                            lam = dual_demand[k_state, i] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                            cutcoef_U[i] += lam                                                 # Coefficient
                            cut_rhs += lam * U_prev[i]                                          # RHS

                        # Inventory-balance constraints
                        for i in range(self.V):
                            for r in range(self.R):
                                lam = dual_inventory[k_state, i, r] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                cutcoef_I[i, r] += lam                                          # Coefficient
                                cut_rhs -= lam * I_prev[i, r]                                   # RHS
                                

                        # Flow capacity constraints
                        for i in range(self.V):
                            for j in range(self.V):
                                if (i, j) in self.A:
                                    lam = dual_capacity[k_state, i, j] * self.transition_dict[str(t-1)][str(n)][str(k_state)]
                                    # Coefficient multiplies fixed parameter K_ijt, so only RHS:
                                    cut_rhs -= lam * self.K_ijt[i][j][t-1]

                # Build the left-hand-side linear expression
                lhs_expr  = self.theta[t-1, n]
                lhs_expr += gp.quicksum(cutcoef_U[i] * self.U[t-1, n][i] for i in range(self.V))
                lhs_expr += -gp.quicksum(cutcoef_I[i, r] * self.I[t-1, n][i, r] for i in range(self.V) for r in range(self.R))
                
                # if t == self.t_L - 1:
                #     print(f"stage{t} cut")
                #     print("cutcoef_U = ")
                #     for i in range(self.V):
                #         if abs(cutcoef_U[i]) > 1e-6:
                #             print(f"{i}, {cutcoef_U[i]}")
                #     print("cutcoef_I = ")
                #     for i in range(self.V):
                #         for r in range(self.R):
                #             if abs(cutcoef_I[i, r]) > 1e-6:
                #                 print(f"{i}, {cutcoef_I[i,r]}")
                
                # if t == self.t_L:
                    # print(f"stage{t} cut")
                    # print("cutcoef_U = ")
                    # for i in range(self.V):
                    #     if abs(cutcoef_U[i]) > 1e-6:
                    #         print(f"{i}, {cutcoef_U[i]}")
                    # print("cutcoef_I = ")
                    # for i in range(self.V):
                    #     for r in range(self.R):
                    #         if abs(cutcoef_I[i, r]) > 1e-6:
                    #             print(f"{i}, {cutcoef_I[i,r]}")
                # exit(0)
                # Add the cut
                self.models[t-1, n].addConstr(lhs_expr >= Qvalue + cut_rhs, name=f"OptCut_{t-1}_{n}_{iter}")

        return cutviolFlag
    
    # Train model
    def solve_sddp(self):

        self.build_models()

        I_ir0 = self.network_processor.I_ir0
        k_init = self.nodeLists[0][0]

        # Set the RHS of the first_stage problem
        for i in range(self.V):
            for r in range(self.R):
                self.inventory_update_constr[0, k_init][i, r].setAttr(GRB.Attr.RHS, I_ir0[i][r])

        # Initialization
        train_time = 0
        relative_gap = self.rel_gap
        lb = 0
        LB = []          # lower-bound history 
        thetaval = [0] * len(self.T_Bef)
        iter = 0
        cutviol_iter = 0
        start = time.time()

        while True:
            iter += 1

            # Forward pass
            if self.travel_time == 1:
                I_prev, U_prev, thetaval, lb, state_path = self.forward_pass()
            elif self.travel_time == 2:
                I_prev, U_prev, f_prev, thetaval, lb, state_path = self.forward_pass()
            
            LB.append(lb)

            # Termination check
            flag, Elapsed = msp_helper.termination_check(iter, relative_gap, LB, start, cutviol_iter, self.params)
            if flag != 0:
                train_time = Elapsed
                break
            
            # Backward pass (if not terminated)
            if self.travel_time == 1:
                cut_violFalg = self.backward_pass_time1(I_prev, U_prev, thetaval, state_path)
            elif self.travel_time == 2:
                cut_violFalg = self.backward_pass(I_prev, U_prev, f_prev, thetaval, state_path)

            # Print outputs for debugging
            print(f"Iteration {iter:>2} | LB: {lb:.2f} | Cut Violated: {cut_violFalg}")

            # cut-violation
            if cut_violFalg:
                cutviol_iter = 0
            else:
                cutviol_iter += 1
        
        # Print for debugging
        # print("\n====== Final Aggregated Profiling Stats ======")
        # for key, val in self.timer_stats.items():
        #     print(f"{key:<30}: {val:.4f} sec")

        return LB, train_time, iter


    
        
 




            


