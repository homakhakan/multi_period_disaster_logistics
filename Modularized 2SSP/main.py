import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import time
from demand_data import DemandData
from network_data import NetworkData
from model import create_ext, master_problem, optimality_sub, feasibility_sub, create_mvp, dvs_mvp, cons_mvp, obj_mvp, compute_mvp_objective
from method import benders
import output

if __name__ == "__main__":
    # Define parameters
    num_sce = 10
    instance_version = "original"
    tperiod = 12
    thetaLB = 0
    eps = 1e-4

    # Create an instance of the DemandData class
    demand_processor = DemandData(num_sce, instance_version)
    # Now call the methods using this instance
    scenario_data = demand_processor.generate_scenario_data()
    demand_processor.load_demand_data(scenario_data["demand_file"], scenario_data["sheet_names"])
    # Extract `P` from `scenario_data`
    P = scenario_data["P"]
    print("Sample demand data for location 5, time 2, scenario 0:", demand_processor.d_its[(7, 2, 3)])
    print("Sample demand data for location 6, time 2, scenario 0:", demand_processor.d_its[(1, 0, 1)])

    # Compute and load expected demand for mvp
    output_demand_file = "expected_demand.xlsx"
    demand_processor.calculate_expected_demand(output_demand_file)
    expected_demand_file = "expected_demand.xlsx"
    d_it = demand_processor.load_expected_demand(expected_demand_file)

    # Print sample values to verify
    print("Sample demand data for location 7, time 2:", d_it[(7, 2)])
    print("Sample demand data for location 36, time 11:", d_it[(36, 11)])

    network_data = NetworkData(instance_version, tperiod)
    sets = network_data.define_sets()
    parameters = network_data.define_parameters()
    # Add `P` to `parameters` so it can be used in the model
    parameters["P"] = P
    # Print the output
    print(f"\nAll counties: {sets['C']}")
    print(f"All locations: {sets['V']}")
    print(f"Warehouse/Supplier locations: {len(sets['V0'])}, {sets['V0']}")
    print(f"Shelter locations: {len(sets['V1'])}, {sets['V1']}")
    print(f"RSA locations: {len(sets['V2'])}, {sets['V2']}")
    print(f"PoD locations: {len(sets['V3'])}, {sets['V3']}")
    print(f"All resources: {sets['R']}")
    print(f"Time periods: {sets['T']}")
    print(f"Time Before Landfall: {sets['T_Bef']}")
    print(f"Time After Landfall: {sets['T_Aft']}")

    start_time_ext = time.time()
    # Create and Solve the Extended Formulation Model
    model = create_ext(sets, parameters, d_its, num_sce)
    model.optimize()

    end_time_ext = time.time()
    total_time_ext = end_time_ext - start_time_ext

    # Print Results
    if model.Status == gp.GRB.OPTIMAL:
        print(f"\nOptimal Objective Value: {model.ObjVal}")
        #  Store instance version, number of scenarios, and objective function value
        output.ext_results(model, instance_version, num_sce, total_time_ext, output_filename="ext_results.xlsx")
    else:
        print("\nModel did not solve to optimality.")

    # Print Execution Time
    print(f"\nTotal Execution Time for extended model: {total_time_ext:.2f} seconds")

    
    # start_time_benders = time.time()
    master_model, I, theta = master_problem(sets, parameters, thetaLB)

    # Create Subproblems
    sub_opt, U_opt, S_opt, f_opt, \
    pi_opt_inventory_balance_U, pi_opt_initial_shortage_U, \
    pi_opt_flow_conservation, pi_opt_arc_capacity = optimality_sub(sets, parameters)

    sub_feas, U_feas, S_feas, f_feas, epsilon_1, epsilon_2, epsilon_3, \
    pi_feas_inventory_balance_U, pi_feas_initial_shortage_U, \
    pi_feas_flow_conservation, pi_feas_arc_capacity = feasibility_sub(sets, parameters)

    # Run Benders Decomposition
    LB, UB, iter_count, Fcuts, Ocuts = benders(
        master_model, I, theta, sub_opt, sub_feas, sets, parameters, d_its, eps, thetaLB, pi_opt_inventory_balance_U, pi_opt_initial_shortage_U, pi_opt_flow_conservation, pi_opt_arc_capacity, pi_feas_inventory_balance_U, pi_feas_initial_shortage_U, pi_feas_flow_conservation, pi_feas_arc_capacity, time_limit=10800

    )
   
    # end_time_benders = time.time()
    # total_time_benders = end_time_benders - start_time_benders

    # Store Benders Decomposition Results
    output.benders_results(instance_version, num_sce, LB, UB, output_filename="benders_results.xlsx")

    # Print Benders Final Results
    print("\nBenders Decomposition Finished:")
    print(f"Final Lower Bound: {LB}")
    print(f"Final Upper Bound: {UB}")
    print(f"Total Iterations: {iter_count}")
    print(f"Feasibility Cuts: {Fcuts}")
    print(f"Optimality Cuts: {Ocuts}")
    # print(f"Total Execution Time for Benders Decomposition: {total_time_benders:.2f} seconds")
    
    # Create MVP Model
    mvp_model = create_mvp()
    I_mvp, U_mvp, S_mvp, f_mvp = dvs_mvp(mvp_model, sets)
    cons_mvp(mvp_model, I_mvp, U_mvp, S_mvp, f_mvp, sets, parameters, d_it)
    obj_mvp(mvp_model, I_mvp, U_mvp, S_mvp, f_mvp, sets, parameters)

    # Start recording execution time
    start_time_mvp = time.time()

    # Optimize the MVP model
    mvp_model.optimize()

    # Extract and display the objective function value if the model is optimal
    if mvp_model.status == GRB.OPTIMAL:
        I_mvp_values = {(i, r, t): I_mvp[i, r, t].x for i in range(sets["V"]) for r in range(sets["R"]) for t in range(sets["T"])}
        print(f"Objective Function Value: {mvp_model.ObjVal}")

    else:
        print("\nMVP Model did not solve to optimality.")

    mvp_obj_value = compute_mvp_objective(sets, parameters, d_its, I_mvp_values, sub_opt, 
    pi_opt_inventory_balance_U, pi_opt_initial_shortage_U, pi_opt_flow_conservation)

    # Print the final MVP objective value
    print(f"\nMean Value Problem (MVP) Objective: {mvp_obj_value}")

    # Record execution time
    end_time_mvp = time.time()
    total_time_mvp = end_time_mvp - start_time_mvp
    print(f"Total Execution Time for mvp: {end_time_mvp - start_time_mvp:.2f} seconds")
    output.mvp_results(instance_version, num_sce, total_time_mvp, mvp_obj_value, output_filename="mvp_results.xlsx")