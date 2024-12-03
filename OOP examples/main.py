from data import ScenarioData, InputParams
from masterProblem import MasterProblem
from subProblem import SubProblem
from benders import BendersDecomposition

if __name__ == "__main__":
    scenarios = [1, 2, 4]
    scenario_data = ScenarioData(scenarios)
    master_problem = MasterProblem()
    sub_problem = SubProblem()
    benders = BendersDecomposition(scenario_data, master_problem, sub_problem)
    benders.run()
