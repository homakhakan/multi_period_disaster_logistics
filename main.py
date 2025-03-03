import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import time
from demand_data import DemandData
from network_data import NetworkData
from model import TwoStageSPModel
from method import benders
import output

if __name__ == "__main__":
    # Define parameters
    num_sce = 5
    instance_version = "original"
    tperiod = 12
    thetaLB = 0
    eps = 1e-5
    time_limit = 3600

    demand_processor = DemandData(num_sce, instance_version) # Create an instance of the DemandData class
    demand_processor.generate_scenario_data() # Now call the methods using this instance
    demand_processor.load_demand_data()
    # Print the output
    print(f"Number of scenarios: {demand_processor.num_sce}")
    print(f"Generated sheet names: {demand_processor.sheet_names}")
    print("Sample demand data for location 7, time 2, scenario 3:", demand_processor.d_its[(7, 2, 3)])
    print("Sample demand data for location 1, time 0, scenario 1:", demand_processor.d_its[(1, 0, 1)])

    network_processor = NetworkData(instance_version, tperiod) # Create an instance of the NetworkData class
    network_processor.define_sets() # Now call the methods using this instance
    network_processor.define_parameters()
    # Print the output
    print(f"\nAll counties: {network_processor.C}")
    print(f"All locations: {network_processor.V}")
    print(f"Warehouse/Supplier locations: {len(network_processor.V0)}, {network_processor.V0}")
    print(f"Shelter locations: {len(network_processor.V1)}, {network_processor.V1}")
    print(f"RSA locations: {len(network_processor.V2)}, {network_processor.V2}")
    print(f"PoD locations: {len(network_processor.V3)}, {network_processor.V3}")
    print(f"All resources: {network_processor.R}")
    print(f"Time periods: {network_processor.T}")
    print(f"Time Before Landfall: {network_processor.T_Bef}")
    print(f"Time After Landfall: {network_processor.T_Aft}")

    ## Solve extended formulation
    ext_model_instance = TwoStageSPModel(network_processor, demand_processor, num_sce, is_mvp=False)
    start_time_ext = time.time()
    ext_model = ext_model_instance.ext_model()
    ext_model.optimize()
    end_time_ext = time.time()
    total_time_ext = end_time_ext - start_time_ext

    # Print Results
    if ext_model.Status == gp.GRB.OPTIMAL:
        print(f"\nOptimal Objective Value: {ext_model.ObjVal}")
        output.ext_results(ext_model, instance_version, num_sce, total_time_ext, output_filename="ext_results.xlsx")
    else:
        print("\nModel did not solve to optimality.")

    print(f"\nTotal Execution Time for extended model: {total_time_ext:.2f} seconds")
    
    ## Solve Benders
    benders_model_instance = TwoStageSPModel(network_processor, demand_processor, num_sce, is_mvp=False)
    # Define master and subproblems
    benders_model_instance.master_problem(thetaLB)
    benders_model_instance.optimality_sub()
    benders_model_instance.feasibility_sub()   
    start_time_benders = time.time() # Start Benders decomposition
    LB, UB, num_iter, num_Fcuts, num_Ocuts = benders(benders_model_instance, eps, thetaLB, time_limit)
    end_time_benders = time.time()
    total_time_benders = end_time_benders - start_time_benders
    output.benders_results(instance_version, num_sce, LB, UB, output_filename="benders_results.xlsx")
    print(f"Total Execution Time for Benders Decomposition: {total_time_benders:.2f} seconds")

    ## Solve mvp
    mvp_model_instance = TwoStageSPModel(network_processor, demand_processor, num_sce, is_mvp=True)
    mvp_model = mvp_model_instance.ext_model()
    start_time_mvp = time.time() # Start recording execution time
    mvp_model.optimize()

    if mvp_model.status == GRB.OPTIMAL: # Extract I_values after optimization
        I_mvp_values = {(i, r, t): mvp_model.getVarByName(f"I_{i}_{r}_{t}").x
                     for i in range(mvp_model_instance.network_processor.V)
                     for r in range(mvp_model_instance.network_processor.R)
                     for t in range(mvp_model_instance.network_processor.T)}
        mvp_model_instance.optimality_sub()
        mvp_obj_value = mvp_model_instance.compute_mvp_objective(I_mvp_values) # Compute MVP objective using I_mvp_values
        print(f"Objective Function Value: {mvp_model.ObjVal}")
    else:
        print("\nMVP Model did not solve to optimality.")
    end_time_mvp = time.time() # Record execution time
    total_time_mvp = end_time_mvp - start_time_mvp
    print(f"Total Execution Time for mvp: {end_time_mvp - start_time_mvp:.2f} seconds")
    output.mvp_results(instance_version, num_sce, total_time_mvp, mvp_obj_value, output_filename="mvp_results.xlsx")