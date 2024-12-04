import gurobipy as gp;
from gurobipy import GRB;
import numpy as np;

class MasterProblem:
    def __init__(self):
        self.model = gp.Model("MasterProblem")
        self.model.setParam("OutputFlag", 0)
        self.x = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=10, obj=0, name="x")
        self.theta = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, obj=1, name="theta")
        self.model.modelSense = GRB.MINIMIZE
        self.model.update()

    def solve(self):
        self.model.optimize()
        return self.x.x, self.theta.x, self.model.objVal

    def add_cut(self, duals, scenarios, num_scenarios):
        """
        Add an optimality cut to the master problem.
        """
        cut_coef = sum(duals) / num_scenarios
        cut_rhs = np.dot(duals, scenarios) / num_scenarios
        self.model.addConstr(self.theta + cut_coef * self.x >= cut_rhs)
        self.model.update()