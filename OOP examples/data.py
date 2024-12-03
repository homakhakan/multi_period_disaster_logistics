import numpy as np
import gurobipy as gp
from gurobipy import GRB

class ScenarioData:
    def __init__(self, scenarios):
        self.scenarios = scenarios
        self.num_scenarios = len(scenarios)

class InputParams:
    def __init__(self, epsilon, max_iter):
        self.epsilon = epsilon
        self.max_iter = max_iter