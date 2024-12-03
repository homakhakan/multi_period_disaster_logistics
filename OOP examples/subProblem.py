import numpy as np
import gurobipy as gp
from gurobipy import GRB

class SubProblem:
    def __init__(self):
        self.model = gp.Model("SubProblem")
        self.model.setParam("OutputFlag", 0)
        self.y1 = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="y1")
        self.y2 = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="y2")
        self.constraint = self.model.addConstr(self.y1 - self.y2 == 0, name="constr")
        self.model.modelSense = GRB.MINIMIZE
        self.model.update()

    def solve(self, rhs):
        self.constraint.setAttr(GRB.Attr.RHS, rhs)
        self.model.optimize()
        return self.model.objVal, self.constraint.pi