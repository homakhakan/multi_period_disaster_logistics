import numpy as np
import gurobipy as gp
from gurobipy import GRB

class BendersDecomposition:
    def __init__(self, scenario_data, master_problem, sub_problem, epsilon=1e-5):
        self.scenario_data = scenario_data
        self.master_problem = master_problem
        self.sub_problem = sub_problem
        self.LB = 0
        self.UB = float("inf")
        self.epsilon = epsilon

    def run(self):
        iteration = 0
        while (self.UB - self.LB) / max(1e-8, abs(self.UB)) > self.epsilon:
            iteration += 1
            xval, thetaval, lb = self.master_problem.solve()
            self.LB = lb
            duals, tempUB = [], 0
            for scenario in self.scenario_data.scenarios:
                sub_obj, dual = self.sub_problem.solve(scenario - xval)
                tempUB += sub_obj / self.scenario_data.num_scenarios
                duals.append(dual)
            self.UB = min(self.UB, tempUB)
            if abs(self.UB - self.LB) / max(1e-8, abs(self.UB)) < self.epsilon:
                break
            self.master_problem.add_cut(duals, tempUB, self.scenario_data.num_scenarios)
            print(f"Iteration {iteration}: LB = {self.LB}, UB = {self.UB}")