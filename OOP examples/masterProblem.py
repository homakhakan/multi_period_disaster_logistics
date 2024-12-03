import numpy as np
import gurobipy as gp
from gurobipy import GRB

class MasterProblem:
    def __init__(self):
        self.model = gp.Model("MasterProblem")
        self.model.setParam("OutputFlag", 0)
        self.x = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=10, name="x")
        self.theta = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="theta")
        self.model.modelSense = GRB.MINIMIZE
        self.model.update()

    def solve(self):
        self.model.optimize()
        return self.x.x, self.theta.x, self.model.objVal

    def add_cut(self, duals, rhs, num_scenarios):
        cut_coef = sum(duals) / num_scenarios
        self.model.addConstr(self.theta + cut_coef * self.x >= rhs)
        self.model.update()