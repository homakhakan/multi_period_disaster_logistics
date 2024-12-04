import gurobipy as gp;
from gurobipy import GRB;
import numpy as np;

class BendersDecomposition:
    def __init__(self, scenario_data, master_problem, sub_problem, epsilon=1e-5):
        self.scenario_data = scenario_data
        self.master_problem = master_problem
        self.sub_problem = sub_problem
        self.LB = 0
        self.UB = 1e10
        self.iteration = 0
        self.epsilon = epsilon

    def run(self):
        while (self.UB - self.LB) / max(abs(self.UB), 1e-8) > self.epsilon:
            self.iteration += 1
    
            # Solve master problem
            xval, thetaval, obj_val = self.master_problem.solve()
            self.LB = obj_val
    
            # Print current iteration, LB, and UB
            print(f"Iteration {self.iteration}: LB = {self.LB:.6f}, UB = {self.UB:.6f}")
    
            # Solve subproblems
            duals = []
            tempUB = 0
            for idx in range(self.scenario_data.num_scenarios):  # Explicit index iteration
                scenario_value = self.scenario_data.scenarios[idx]  # Access by index
                rhs = scenario_value - xval
                sub_obj, dual = self.sub_problem.solve(rhs)
                tempUB += sub_obj / self.scenario_data.num_scenarios
                duals.append(dual)
    
            # Update UB
            if tempUB < self.UB:
                self.UB = tempUB
    
            # Check convergence
            if (self.UB - self.LB) / max(abs(self.UB), 1e-8) < self.epsilon:
                break
    
            # Add optimality cut
            self.master_problem.add_cut(duals, self.scenario_data.scenarios, self.scenario_data.num_scenarios)
            print(f"cut[{self.iteration}]: cutcoef = {sum(duals) / self.scenario_data.num_scenarios:.6f}, "
                  f"cutrhs = {np.dot(duals, self.scenario_data.scenarios) / self.scenario_data.num_scenarios:.6f}")
    
        print("#########################################################################################")
        print(f"Final LB = {self.LB:.6f}")
        print(f"Final UB = {self.UB:.6f}")
        print(f"xval = {xval}")
