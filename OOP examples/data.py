import gurobipy as gp;
from gurobipy import GRB;
import numpy as np;

class ScenarioData:
    def __init__(self, scenarios):
        self.scenarios = scenarios
        self.num_scenarios = len(scenarios)