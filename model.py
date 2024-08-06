import gurobipy as gp
from gurobipy import GRB
import pandas as pd

class OptimizationModel:
    def __init__(self, model_setup):
        self.model_setup = model_setup
        self.model = None
        self.f = {}
        self.I = {}
        self.J = {}
        self.K = {}
        self.U = {}
        self.S = {}
        self.b = None

    def define_model(self):
        self.model = gp.Model("CVRP")
        self.model.modelSense = GRB.MINIMIZE

    def add_variables(self):
        # Define decision variables
        for i in range(self.model_setup.V):
            for j in range(self.model_setup.V):
                for r in range(self.model_setup.R):
                    for t in range(self.model_setup.T):
                        if (i, j) in self.model_setup.A:
                            self.f[i, j, r, t] = self.model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="f_%s_%s_%s_%s" % (i, j, r, t))

        for i in range(self.model_setup.V):
            for r in range(self.model_setup.R):
                for t in range(self.model_setup.T):
                    self.I[i, r, t] = self.model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="I_%s_%s_%s" % (i, r, t))

        for i in range(self.model_setup.V):
            for t in range(self.model_setup.T):
                self.J[i, t] = self.model.addVar(vtype=GRB.INTEGER, obj=0, name="J_%s_%s" % (i, t))

        for i in range(self.model_setup.V):
            for j in range(self.model_setup.V):
                for r in range(self.model_setup.R):
                    for t in range(self.model_setup.T):
                        if (i, j) in self.model_setup.A:
                            self.K[i, j, r, t] = self.model.addVar(vtype=GRB.INTEGER, obj=0, name="K_%s_%s_%s_%s" % (i, j, r, t))

        for i in range(self.model_setup.V):
            for t in range(self.model_setup.T):
                self.U[i, t] = self.model.addVar(vtype=GRB.CONTINUOUS, obj=self.model_setup.params['p_i'][i] * self.model_setup.tperiod * self.model_setup.weight_penalty, name="U_%s_%s" % (i, t))

        for i in range(self.model_setup.V):
            for t in range(self.model_setup.T):
                self.S[i, t] = self.model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="S_%s_%s" % (i, t))

        self.b = self.model.addVar(vtype=GRB.CONTINUOUS, obj=self.model_setup.weight_cost, name="b")

    def add_constraints(self):
        V1 = self.model_setup.V1
        V3 = self.model_setup.V3
        T = self.model_setup.T
        A = self.model_setup.A
        d_shelter = self.model_setup.params['d_shelter']
        d_pod = self.model_setup.params['d_pod']
        I_ir0 = self.model_setup.params['I_ir0']
        t_ijt = self.model_setup.params['t_ijt']
        g_shelter = self.model_setup.params['g_shelter']
        g_pod = self.model_setup.params['g_pod']
        h_i = self.model_setup.params['h_i']
        w_r = self.model_setup.params['w_r']
        m_ij = self.model_setup.params['m_ij']
        cost_ir = self.model_setup.params['cost_ir']

        # Shortage and Demand Satisfaction Constraints
        for i in V1:
            self.model.addConstr(self.U[i, 0] == d_shelter[i - V1[0]][0])
        for i in V3:
            self.model.addConstr(self.U[i, 0] == d_pod[i - V3[0]][0])
        for i in V1:
            for t in range(T):
                if t > 0:
                    self.model.addConstr(self.U[i, t] == self.U[i, t - 1] - self.S[i, t] + d_shelter[i - V1[0]][t])
        for i in V3:
            for t in range(T):
                if t > 0:
                    self.model.addConstr(self.U[i, t] == self.U[i, t - 1] - self.S[i, t] + d_pod[i - V3[0]][t])
        for i in range(self.model_setup.V):
            self.model.addConstr(self.S[i, 0] == 0)
            for t in range(T):
                if t > 0:
                    self.model.addConstr(self.S[i, t] <= self.U[i, t - 1])

        # Starting Value of I[t=0]
        for i in range(self.model_setup.V):
            for r in range(self.model_setup.R):
                self.model.addConstr(self.I[i, r, 0] == I_ir0[i][r])

        # Inventory Flow Balance Constraints
        for r in range(self.model_setup.R):
            for t in range(T):
                if t <= T - 2:
                    for i in V1:
                        self.model.addConstr(
                            self.I[i, r, t + 1] == self.I[i, r, t]
                            - sum(self.f[i, j, r, t] for j in range(self.model_setup.V) if (i, j) in A)
                            + sum(self.f[j, i, r, (t - t_ijt[j][i] + 1)] for j in range(self.model_setup.V) if (j, i) in A and (t - t_ijt[j][i] + 1) >= 0)
                            - self.S[i, t] * g_shelter[r]
                        )
                    for i in V3:
                        self.model.addConstr(
                            self.I[i, r, t + 1] == self.I[i, r, t]
                            - sum(self.f[i, j, r, t] for j in range(self.model_setup.V) if (i, j) in A)
                            + sum(self.f[j, i, r, (t - t_ijt[j][i] + 1)] for j in range(self.model_setup.V) if (j, i) in A and (t - t_ijt[j][i] + 1) >= 0)
                            - self.S[i, t] * g_pod[r]
                        )
                    for i in self.model_setup.V0:
                        self.model.addConstr(
                            self.I[i, r, t + 1] == self.I[i, r, t]
                            - sum(self.f[i, j, r, t] for j in range(self.model_setup.V) if (i, j) in A)
                            + sum(self.f[j, i, r, (t - t_ijt[j][i] + 1)] for j in range(self.model_setup.V) if (j, i) in A and (t - t_ijt[j][i] + 1) >= 0)
                        )
                    for i in self.model_setup.V2:
                        self.model.addConstr(
                            self.I[i, r, t + 1] == self.I[i, r, t]
                            - sum(self.f[i, j, r, t] for j in range(self.model_setup.V) if (i, j) in A)
                            + sum(self.f[j, i, r, (t - t_ijt[j][i] + 1)] for j in range(self.model_setup.V) if (j, i) in A and (t - t_ijt[j][i] + 1) >= 0)
                        )
                for i in V1:
                    self.model.addConstr(self.I[i, r, t] >= sum(self.f[i, j, r, t] for j in range(self.model_setup.V) if (i, j) in A) + self.S[i, t] * g_shelter[r])
                for i in V3:
                    self.model.addConstr(self.I[i, r, t] >= sum(self.f[i, j, r, t] for j in range(self.model_setup.V) if (i, j) in A) + self.S[i, t] * g_pod[r])
                for i in self.model_setup.V0:
                    self.model.addConstr(self.I[i, r, t] >= sum(self.f[i, j, r, t] for j in range(self.model_setup.V) if (i, j) in A))
                for i in self.model_setup.V2:
                    self.model.addConstr(self.I[i, r, t] >= sum(self.f[i, j, r, t] for j in range(self.model_setup.V) if (i, j) in A))

        # Starting Value of J[t=0]
        for i in range(self.model_setup.V):
            self.model.addConstr(self.J[i, 0] == self.model_setup.params['j_i0'][i])

        # Truck Flow Balance Constraints
        for i in range(self.model_setup.V):
            for t in range(T):
                if t < T - 1:
                    self.model.addConstr(
                        self.J[i, t + 1] == self.J[i, t]
                        - sum(sum(self.K[i, j, r, t] for j in range(self.model_setup.V) if (i, j) in A) for r in range(self.model_setup.R))
                        + sum(sum(self.K[j, i, r, (t - t_ijt[j][i] + 1 - h_i[i])] for j in range(self.model_setup.V) if (j, i) in A and (t - t_ijt[j][i] + 1 - h_i[i]) >= 0) for r in range(self.model_setup.R))
                    )
                self.model.addConstr(self.J[i, t] >= sum(sum(self.K[i, j, r, t] for j in range(self.model_setup.V) if (i, j) in A) for r in range(self.model_setup.R)))

        # Capacity of Truck Constraints
        for i in range(self.model_setup.V):
            for j in range(self.model_setup.V):
                for r in range(self.model_setup.R):
                    for t in range(self.model_setup.T):
                        if (i, j) in A:
                            self.model.addConstr(w_r[r] * self.f[i, j, r, t] <= self.K[i, j, r, t])

        # Budget Constraint
        self.model.addConstr(
            gp.quicksum(
                m_ij[i][j] * self.K[i, j, r, t]
                for i in range(self.model_setup.V)
                for j in range(self.model_setup.V)
                for r in range(self.model_setup.R)
                for t in range(self.model_setup.T)
                if (i, j) in self.model_setup.A
            ) + gp.quicksum(
                cost_ir[i][r] * (self.I[i, r, 0] - self.I[i, r, self.model_setup.T - 1])
                for i in self.model_setup.V0
                for r in range(self.model_setup.R)
            ) <= self.b
        )

    def optimize(self):
        self.model.update()
        self.model.setParam("TimeLimit", 2 * 3600)
        self.model.setParam("MIPGap", 0.01)
        self.model.optimize()
