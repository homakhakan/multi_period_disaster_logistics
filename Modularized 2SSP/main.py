import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import time
from demand_data import DemandData
from network_data import NetworkData
from model import TwoStageSPModel
from method import benders, solve_rolling_horizon
import output
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", "--num_sce", type = int, help="Number of scenarios")
    parser.add_argument("-iv", "--instance_version", type=str, choices=["smallsize", "mediumsize", "largesize", "original"], help="Instance version: 0. original, 1. smallsize, 2. mediumsize, 3. largesize")
    parser.add_argument("-tl", "--time_limit", type=int, help="Time limit in seconds for running Benders")
    parser.add_argument("-sm", "--solution_method", type=int, choices=[1, 2, 3, 4], required=True, help="Solution method: 1. extended, 2. benders, 3. mvp, 4.RH")
    args = parser.parse_args()

    with open("fixed_params.yaml", "r") as f:
        params = yaml.safe_load(f)
        tperiod = int(params['TPERIOD'])
        thetaLB = float(params['THETA_LB'])
        eps = float(params['EPS'])

    demand_processor = DemandData(args.num_sce, args.instance_version) # Create an instance of the DemandData class
    demand_processor.generate_scenario_data() # Now call the methods using this instance
    demand_processor.load_demand_data()
    # Print the output
    print(f"Number of scenarios: {demand_processor.num_sce}")
    print(f"Generated sheet names: {demand_processor.sheet_names}")
    #print("Sample demand data for location 7, time 2, scenario 3:", demand_processor.d_its[(6, 2, 4)])

    network_processor = NetworkData(args.instance_version, tperiod) # Create an instance of the NetworkData class
    network_processor.define_sets() # Now call the methods using this instance
    network_processor.define_parameters()

    print("\nReading data...")  

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
    print(f"Scenario indices: {demand_processor.sce}")
    print(f"Scenario probabilities: {demand_processor.P}")

    # Run only the selected model
    if args.solution_method == 1:
        ext_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce)
        start_time_ext = time.time()
        ext_model = ext_model_instance.ext_model()
        ext_model.optimize()
        end_time_ext = time.time()
        total_time_ext = end_time_ext - start_time_ext

        # Print and save results
        if ext_model.Status == gp.GRB.OPTIMAL:
            print(f"\nOptimal Objective Value: {ext_model.ObjVal}")
            output.ext_results(ext_model_instance, ext_model, args.instance_version, args.num_sce, end_time_ext - start_time_ext, network_processor.multiplier1, network_processor.multiplier2, network_processor.multiplier3, output_filename="ext_results.xlsx")
        else:
            print("\nModel did not solve to optimality.")

        print(f"Total Execution Time for Extended Model: {total_time_ext:.2f} seconds")

    elif args.solution_method == 2:
        benders_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce)
        benders_model_instance.master_problem(thetaLB)
        benders_model_instance.optimality_sub()
        benders_model_instance.feasibility_sub()
        start_time_benders = time.time()
        LB, UB, num_iter, num_Fcuts, num_Ocuts = benders(benders_model_instance, eps, thetaLB, args.time_limit)
        end_time_benders = time.time()
        total_time_benders = end_time_benders - start_time_benders

        output.benders_results(args.instance_version, args.num_sce, total_time_benders, LB, UB, output_filename="benders_results.xlsx")
        print(f"Total Execution Time for Benders: {total_time_benders:.2f} seconds")

    elif args.solution_method == 3:
        demand_processor.compute_expected_demand()
        mvp_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce)
        mvp_model = mvp_model_instance.ext_model(is_mvp=True)
        start_time_mvp = time.time()
        mvp_model.optimize()
        print(f"\n MVP Model Objective Value: {mvp_model.ObjVal}")

        if mvp_model.status == gp.GRB.OPTIMAL:
            I_mvp_values = {(i, r, t): mvp_model.getVarByName(f"I_{i}_{r}_{t}").x
                            for i in range(mvp_model_instance.network_processor.V)
                            for r in range(mvp_model_instance.network_processor.R)
                            for t in range(mvp_model_instance.network_processor.T)}

            # Restore full scenario-based demand and probabilities
            mvp_model_instance.demand_processor.d_its = mvp_model_instance.demand_processor.d_its_original
            mvp_model_instance.demand_processor.P = mvp_model_instance.demand_processor.P_original
            mvp_model_instance.num_sce = len(mvp_model_instance.demand_processor.P_original)
            mvp_model_instance.optimality_sub()
            mvp_obj_value = mvp_model_instance.compute_mvp_objective(I_mvp_values)
            print(f"MVP Objective Function Value: {mvp_model.ObjVal}")
        else:
            print("\nMVP Model did not solve to optimality.")

        end_time_mvp = time.time()
        total_time_mvp = end_time_mvp - start_time_mvp
        print(f"Total Execution Time for MVP: {total_time_mvp:.2f} seconds")
        output.mvp_results(args.instance_version, args.num_sce, total_time_mvp, mvp_obj_value, network_processor.multiplier1, network_processor.multiplier2, network_processor.multiplier3, output_filename="mvp_results.xlsx")
