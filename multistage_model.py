import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import msp_helper

class MultiStageSPModel:
    def __init__(self, network_processor, demand_processor):
        self.network_processor = network_processor
        self.demand_processor = demand_processor

        self.V = network_processor.V
        self.V1 = network_processor.V1
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
        self.delta = {}
        for (i, j) in self.A:
            for t in range(1, self.T):
                t_prime = t - self.t_ijt[i][j] - 1
                if 1 <= t_prime < t:             
                    self.delta[(i, j, t, t_prime)] = 1

        # Markov chain data
        transition_path = "Data/pi_mssp_d_ST_track_15.json"
        self.transition_dict = msp_helper.load_transition_dict(transition_path)
        self.nodeLists = msp_helper.build_node_lists(self.transition_dict)


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

        f1 = {}
        for i in range(self.V):
            for j in range(self.V):
                if (i, j) in self.A:
                    for r in range(self.R):
                        for t_prime in range(1, t):
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
            theta = m.addVar(name="theta")
        else:
            theta = None

        # Constraints
        # Demand shortage update
        demand_shortage_constr = {}
        for i in range(self.V):
            demand_shortage_constr[i] = m.addConstr(U[i] + S[i] == 0, name=f"DemandShortageUpdate_{i}")

        # Inventory update
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

                inventory_update_constr[i, r] = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) + S[i] * g_coeff == 0, name=f"InventoryUpdate_{i}_{r}")

        # Flow capacity
        flow_capacity_constr = {}
        for i in range(self.V):
            for j in range(self.V):
                if (i, j) in self.A:
                    cap_ij_t = self.K_ijt[i][j][t]
                    flow_capacity_constr[i, j] = m.addConstr(gp.quicksum(self.w_r[r] * f[i, j, r] for r in range(self.R)) <= cap_ij_t, name=f"FlowCapacity_{i}_{j}_{t}")

        # In-transit flow update to the next stage
        in_transit_update_constr = {}
        for i in range(self.V):
            for j in range(self.V):
                if (i, j) in self.A:
                    for r in range(self.R):
                        # t′ < t: Carry forward past flows (to be updated later)
                        for t_prime in range(1, t):
                            in_transit_update_constr[i, j, r, t_prime] = m.addConstr(f1[i, j, r, t_prime] == 0, name=f"CarryTransit_{i}_{j}_{r}_{t_prime}_to_{t}")

                        # t′ = t: Assign new flow to f1 for the next stage
                        in_transit_update_constr[i, j, r, t] = m.addConstr(f1[i, j, r, t] == f[i, j, r], name=f"InTransitNew_{i}_{j}_{r}_{t}")

        # Objective
        objective = (
            gp.quicksum(self.p_i[i] * U[i] for i in range(self.V)) 
            + gp.quicksum(
                gp.quicksum(self.cost_ir[i][r] * f[i, j, r] for i in self.V0 for j in range(self.V) if (i, j) in self.A)
                + gp.quicksum(self.m_ij[i][j] * f[i, j, r] for i in range(self.V) for j in range(self.V) if (i, j) in self.A) for r in range(self.R))
            + (theta if not last_stage else 0)
        )

        m.setObjective(objective, GRB.MINIMIZE)

        return {
        "model": m,
        "I": I,                    # state variable: inventory
        "U": U,                    # state variable: unmet demand
        "f1": f1,                  # state variable: in-transit flow
        "f": f,                    # local variable: flow
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
        self.f1 = {}
        self.f = {}
        self.S = {}
        self.theta = {}
        self.demand_shortage_constr = {}
        self.inventory_update_constr = {}
        self.flow_capacity_constr = {}
        self.in_transit_update_constr = {}

        # Loop over pre-landfall stochastic stages (t = 1 … t_L)
        for t in range(1, len(self.T_Bef)):
            for k in self.nodeLists[t]: # Every reachable Markov state at time t
                last_stage = (t == self.t_L) # True only at t = t_L

                # Build the stage-state model and unpack everything we need
                model_dict = self.stage_t_state_k_problem(t, k, last_stage=last_stage)

                self.models[t, k] = model_dict["model"]
                self.I[t, k] = model_dict["I"]
                self.U[t, k] = model_dict["U"]
                self.f1[t, k] = model_dict["f1"]
                self.f[t, k] = model_dict["f"]
                self.S[t, k] = model_dict["S"]
                self.theta[t, k] = model_dict["theta"]
                self.demand_shortage_constr[t, k] = model_dict["demand_shortage_constr"]
                self.inventory_update_constr[t, k] = model_dict["inventory_update_constr"]
                self.flow_capacity_constr[t, k] = model_dict["flow_capacity_constr"]
                self.in_transit_update_constr[t, k] = model_dict["in_transit_update_constr"]

    def forward_pass(self):

        I_ir0 = self.network_processor.I_ir0
        k_t = self.nodeLists[0][0]      # First state at t=0
        state_path = [k_t]              # Keep track of visited states

        # Initialize the state variables from t=0
        I_prev = {(i, r): I_ir0[i][r] for i in range(self.V) for r in range(self.R)}
        U_prev = {i: 0 for i in range(self.V)}
        f_prev = {} # No past flows exist before t=1

        for t in range(1, len(self.T_Bef)):
            k_t = msp_helper.MC_sample(t - 1, k_t, self.transition_dict)
            state_path.append(k_t)          # Keep sample path history
            self.mspFP_update_rhs(t, k_t, I_prev, U_prev, f_prev) # Update the right hand side of the constraints
            model_dict = self.models[t, k_t]
            model = model_dict["model"]
            model.optimize() # Optimize the model

            # Check for the optimality and then extract the solution for updating state variables
            if model.status != gp.GRB.OPTIMAL:
                raise RuntimeError(f"Model at t={t}, state={k_t} not optimal")
            else:
                I_prev = {(i, r): model_dict["I"][i, r].X for i in range(self.V) for r in range(self.R)} # Extract inventory level (state var)
                U_prev = {i: model_dict["U"][i].X for i in range(self.V)} # Extract demand shortage (state var)

                # Extract the new in-transit flow for next stage
                f_current = {}

                # Carry forward earlier shipments t' < t that have NOT arrived
                for (i, j) in self.A:
                    for r in range(self.R):
                        for t_prime in range(1, t):              # Only past departures
                            if self.delta.get((i, j, t, t_prime), 0) == 0:
                                # still travelling, keep it
                                f_current[(i, j, r, t_prime)] = f_prev.get((i, j, r, t_prime), 0.0)
                
                # Add NEW shipments that depart at the current stage t
                for (i, j, r), var in model_dict["f"].items():  # local decision variable
                    if var.X > 1e-6:
                        f_current[(i, j, r, t)] = var.X
    
                # Save it to use as f_prev when we move to stage t+1
                f_prev = f_current

                ## Alternative for updating f_prev
                # f_prev = {key: var.X for key, var in model_dict["f1"].items() if var.X > 1e-8}

        return I_prev, U_prev, f_prev, state_path

    def mspFP_update_rhs(self, t, k_t, I_prev, U_prev, f_prev):
        """Update right-hand sides of the (t , k_t) model constraints before optimization.""" 

        # Realized demand at stage t, given Markov state k_t
        demand_real = self.demand_processor.msp_demand_mapping(t, k_t)
    
        # Demand shortage rhs
        for i, cons in self.demand_shortage_constr[t, k_t].items():
            rhs = U_prev.get(i, 0) + demand_real.get(i, 0)
            cons.setAttr(GRB.Attr.RHS, rhs)

        # Inventory flow-balance rhs
        for (i, r), cons in self.inventory_update_constr[t, k_t].items():
            arrivals = 0
            for j in range(self.V):
                if (j, i) in self.A:
                    for t_prime in range(1, t):
                        if self.delta.get((j, i, t, t_prime), 0) == 1:
                            arrivals += f_prev.get((j, i, r, t_prime), 0)
            rhs = I_prev.get((i, r), 0) + arrivals
            cons.setAttr(GRB.Attr.RHS, rhs)
        
        # In-transit flow rhs
        for (i, j, r, t_prime), cons in self.in_transit_update_constr[t, k_t].items():
            if self.delta.get((i, j, t, t_prime), 0) == 0:
                rhs = f_prev.get((i, j, r, t_prime), 0)
            else:
                rhs = 0
            cons.setAttr(GRB.Attr.RHS, rhs)

    def backward_pass(self, I_prev, U_prev, f_prev, state_path):

        for t in range(len(state_path)-1, 0, -1):
            Q = {}  # Use dictionary to store: key = Markov state (tuple), value = objVal
            # List for all the dual multipliers of the constraints
            dual_demand    = {}  # (k,i)              
            dual_inventory = {}  # (k,i,r)            
            dual_capacity  = {}  # (k,i,j)            
            dual_transit   = {}  # (k,i,j,r,t′)    

            prev_state = state_path[t - 1]  # The state observed at time t-1

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
                                    for t_prime in range(1, t):
                                        dual_transit[k_state, i, j, r, t_prime] = self.in_transit_update_constr[t, k_state][i, j, r, t_prime].pi

            # Solving all stage-(t-1) problems and generate cuts/valid inequalities
            for n in self.nodeLists[t - 1]:
                Qvalue = 0
                for k_state in self.nodeLists[t]:
                    if 




            


