import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

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

        # Builds models for each stage t before landfall
        self.stage = [self.stage_problem(t) for t in range(1, len(self.T_Bef) - 1)]
        # Append the last stochastic stage with theta removed
        self.stage.append(self.stage_problem(len(self.T_Bef) - 1, last_stage=True)) 

        # Define delta indicator
        self.delta = {}
        for (i, j) in self.A:
            for t in range(1, self.T):
                t_prime = t - self.t_ijt[i][j] - 1
                if 1 <= t_prime < t:             # valid stage
                    self.delta[(i, j, t, t_prime)] = 1


    def stage_problem(self, t, last_stage=False):
    
        # Create a model for each stage t
        m = gp.Model(f"Stage_{t}_model")
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

        # Define local variables
        S = {}
        for i in range(self.V):
            S[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{i}")

        f = {}      
        for i in range(self.V):
            for j in range(self.V):
                if (i, j) in self.A:
                    for r in range(self.R):
                        f[i, j, r] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_{i}_{j}_{r}")

        # We need to define the last-stage problem separately since we do not have the theta variable there
        if not last_stage:
            theta = m.addVar(name="theta")
        else:
            theta = None

        # Constraints
        constraints = []

        # Demand shortage update
        demand_shortage_constr = {}
        for i in range(self.V):
            cons = m.addConstr(U[i] + S[i] == 0, name=f"DemandShortageUpdate_{i}")
            demand_shortage_constr[i] = cons
            constraints.append(cons)

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

                cons = m.addConstr(I[i, r] + gp.quicksum(f[i, j, r] for j in range(self.V) if (i, j) in self.A) + S[i] * g_coeff == 0, name=f"InventoryUpdate_{i}_{r}")
                inventory_update_constr[i, r] = cons
                constraints.append(cons)

        # Flow capacity
        flow_capacity_constr = {}
        for i in range(self.V):
            for j in range(self.V):
                if (i, j) in self.A:
                    cap_ij_t = self.K_ijt[i][j][t]
                    cons = m.addConstr(gp.quicksum(self.w_r[r] * f[i, j, r] for r in range(self.R)) <= cap_ij_t, name=f"FlowCapacity_{i}_{j}_{t}")
                    flow_capacity_constr[i, j] = cons
                    constraints.append(cons)

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
        "S": S,                    # local variable
        "f": f,                    # local variable: flow
        "theta": theta,            # cost-to-go (None in last stage)
        "constraints": constraints,
        "demand_shortage_constr": demand_shortage_constr,
        "inventory_update_constr": inventory_update_constr,
        "flow_capacity_constr": flow_capacity_constr
        }
    
    def forward_pass(self):

        # Demand realization part



        I_ir0 = self.network_processor.I_ir0

        # Initialize the state variables from t=0
        I_prev = {(i, r): I_ir0[i][r] for i in range(self.V) for r in range(self.R)}
        U_prev = {i: 0.0 for i in range(self.V)}
        f_prev = {} # No past flows exist before t=1

        for t in range(1, len(self.T_Bef)):
            model_data = self.stage[t - 1] # Subtract 1 since self.stage[0] corresponds to time step t = 1 (not t = 0)

            # Retrieve specific constraints for RHS updates
            demand_constr = model_data["demand_shortage_constr"]
            inv_constr = model_data["inventory_update_constr"]

            # Update RHS for each constraint directly
            # Update RHS of demand shortage constraint
            for i, cons in demand_constr.items():
                rhs = U_prev.get(i, 0) + "demand_realized[i]" ### Demand to be specified later 
                cons.setAttr(GRB.Attr.RHS, rhs)

            # Update RHS of inventory balance constraint
            for (i, r), cons in inv_constr.items():
                arrivals = 0
                for j in range(self.V):                      
                    if (j, i) in self.A:                
                        for t_prime in range(1, t):         
                            if self.delta.get((j, i, t, t_prime), 0) == 1:
                                arrivals += f_prev.get((j, i, r, t_prime), 0)
                rhs = I_prev.get((i, r), 0) + arrivals
                cons.setAttr(GRB.Attr.RHS, rhs)

            # Optimize the model
            model_data["model"].update()
            model_data["model"].optimize()

            # Check for the optimality and then extract the solution for updating state variables
            if model_data["model"].status != GRB.OPTIMAL:
                raise RuntimeError("Something is wrong in the forward pass!")
            else:
                I_prev = {(i, r): model_data["I"][i, r].X for i in range(self.V) for r in range(self.R)} # Extract inventory level (state var)
                U_prev = {i: model_data["U"][i].X for i in range(self.V)} # Extract demand shortage (state var)
            


