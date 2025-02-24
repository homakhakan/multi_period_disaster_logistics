import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import time
from demand_data import DemandData
from network_data import NetworkData
from model import create_ext
import output

if __name__ == "__main__":
    # Define parameters
    num_sce = 10
    instance_version = "original"
    tperiod = 12
    
    start_time = time.time()

    # Create an instance of the DemandData class
    demand_processor = DemandData(num_sce, instance_version)
    # Now call the methods using this instance
    scenario_data = demand_processor.generate_scenario_data()
    d_its = demand_processor.load_demand_data(scenario_data["demand_file"], scenario_data["sheet_names"])
    # 🔹 Extract `P` from `scenario_data`
    P = scenario_data["P"]
    print("Sample demand data for location 5, time 2, scenario 0:", d_its[(7, 2, 3)])
    print("Sample demand data for location 6, time 2, scenario 0:", d_its[(1, 0, 1)])

    
    network_data = NetworkData(instance_version, tperiod)
    sets = network_data.define_sets()
    parameters = network_data.define_parameters()
    # ✅ Add `P` to `parameters` so it can be used in the model
    parameters["P"] = P
    # Print the output
    print(f"\n All counties: {sets['C']}")
    print(f" All locations: {sets['V']}")
    print(f" Warehouse/Supplier locations: {len(sets['V0'])}, {sets['V0']}")
    print(f" Shelter locations: {len(sets['V1'])}, {sets['V1']}")
    print(f" RSA locations: {len(sets['V2'])}, {sets['V2']}")
    print(f" PoD locations: {len(sets['V3'])}, {sets['V3']}")
    print(f" All resources: {sets['R']}")
    print(f" Time periods: {sets['T']}")
    print(f" Time Before Landfall: {sets['T_Bef']}")
    print(f" Time After Landfall: {sets['T_Aft']}")

    # Create and Solve the Extended Formulation Model
    model = create_ext(sets, parameters, d_its, num_sce)
    model.optimize()

    end_time = time.time()

    # Print Results
    if model.Status == gp.GRB.OPTIMAL:
        print(f"\n Optimal Objective Value: {model.ObjVal}")
        #  Store instance version, number of scenarios, and objective function value
        output.store_objective_value(model, instance_version, num_sce, output_filename="solution_summary.xlsx")
    else:
        print("\n⚠ Model did not solve to optimality.")

    # Print Execution Time
    total_time = end_time - start_time
    print(f"\n⏳ Total Execution Time for extended model: {total_time:.2f} seconds")