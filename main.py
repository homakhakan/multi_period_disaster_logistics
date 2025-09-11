import os
import sys
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import time
import argparse
import yaml
from shapely.geometry import Point, LineString
import math

# === Setup to use Sudhan's repo ===
#test
# Step 1: Define repo path (relative)
friend_dir = os.path.join(os.path.dirname(__file__), "MSSP_IHRLEP_Networks")

# Step 2: Add it to sys.path to allow imports
sys.path.append(friend_dir)

# Step 3: Temporarily set working directory to Sudhan's repo
os.chdir(friend_dir)

# Step 4: Import from external code
from data_main import Data
from demand_generation import in_sample, demand_deterministic_landfall, data
import commands
import helper

# Step 5: Switch back to original repo root
os.chdir(os.path.dirname(__file__))

# === My own project modules ===
from demand_data import DemandData
from network_data import NetworkData
from twostage_model import TwoStageSPModel
import output
from output import save_full_oos_demand

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", "--num_sce", type = int, help="Number of scenarios")
    parser.add_argument("-iv", "--instance_version", type=str, choices=["original"])
    parser.add_argument("-tl", "--time_limit", type=int, help="Time limit in seconds for running Benders")
    parser.add_argument("-no", "--nbOS", type=int, default=5, help="Number of sample paths for rolling horizon evaluation")
    parser.add_argument("-oos", "--num_oos", type=int, help="Number of out-of-sample scenarios for 2SSP evaluation")
    parser.add_argument("-eos", "--eval_oos", action="store_true", help="Run out-of-sample evaluation for 2SSP after solving extended model")
    parser.add_argument("-sm", "--solution_method", type=int, choices=[1, 2, 3, 4, 5, 6], required=True, help="Solution method: 1. extended, 2. benders, 3. mvp, 4.RH, 5.RH(parallel), 6.multi-stage (SDDP)")
    parser.add_argument("-m", "--model", type=int, choices=[1, 2], required=True, help="Model version: 1 = Using fixed arc capacity, 2 = with vehicle routing decisions")
    parser.add_argument("-tt", "--travel_time", type=int, choices=[1, 2], help="Travel-time mode: 1. All arcs take 1 period to travel, 2. Some arcs might take more than 1 period to travel")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")

    # New args for running the rolling horizon more easily on Palmetto (Run simultaneously)
    parser.add_argument("-si", "--start_index", type=int, default=0, help="Row index in the CSV where this job should start reading")
    parser.add_argument("-np", "--num_paths",   type=int, default=2, help="How many sample-path rows to read (chunk size)")

    args = parser.parse_args()

    with open("fixed_params.yaml", "r") as f:
        params = yaml.safe_load(f)

    thetaLB = float(params["THETA_LB"])
    eps = float(params["EPS"])
    network_processor = NetworkData(args.instance_version, params) # Create an instance of the NetworkData class
    network_processor.define_sets() # Now call the methods using this instance
    network_processor.define_parameters(args.travel_time)

    # Debugging
    # print("Number of arcs in network:", len(network_processor.A))
    # print("Arc capacities K_ijt: max =", np.max(network_processor.K_ijt), ", min =", np.min(network_processor.K_ijt))
    
    # # Debugging
    # print("Travel cost matrix (m_ij): min =", np.min(network_processor.m_ij), ", max =", np.max(network_processor.m_ij))
    # print("Penalty cost per location (p_i):")
    # for i, p in enumerate(network_processor.p_i):
    #     print(f"Location {i}: Penalty = {p}")

    # df_m_ij = pd.DataFrame(
    #     network_processor.m_ij,
    #     columns=[f"To_{j}" for j in range(len(network_processor.df_V))]
    # )
    # df_m_ij.index = [f"From_{i}" for i in range(len(network_processor.df_V))]

    # print("Travel Cost Matrix (m_ij):")
    # print(df_m_ij)

    # # Optional: Save to Excel for inspection
    # df_m_ij.to_excel("travel_cost_matrix.xlsx")

    demand_processor = DemandData(args.num_sce, args.instance_version, args.nbOS, params) # Create an instance of the DemandData class

    # ---------- Simulated In-Sample Demand Setup for static 2SSP-----------
    # Create different scenarios for solving static 2SSP. Based on specifying -ns (number of scenarios), we can have that many scenarios created in "demand_data_dynamic"
    # print("\nGenerating in-sample demand...")
    demand_processor.load_input_files(population_file="Data/PoD location information.xlsx", distance_file="Data/Distance_podtoshelter.xlsx", network_processor=network_processor)
    
    if args.solution_method != 6:
        if args.debug:
            demand_processor.in_sample_data = demand_processor.generate_in_sample_demand(args.num_sce, network_processor, output_dir="Generated data/csv_by_location", debug_mode=True)
            demand_processor.allocate_demand(demand_files = [f"Generated data/csv_by_location/location_{i}.csv" for i in range(network_processor.num_pods)], output_file = "Generated data/Shelter_Demand.xlsx", network_processor=network_processor, debug_mode=True)
            demand_processor.calculate_landfall_demand(wind_speed_file="Generated data/LandfallStates.xlsx", pod_info_file="Data/PoD location information.xlsx", output_file="Generated data/POD_Demand_total.xlsx", debug_mode=True)
            demand_processor.distribute_demand_by_time(total_demand_file="Generated data/PoD_Demand_total.xlsx", network_processor=network_processor, output_file="Generated data/POD_Demand.xlsx", debug_mode=True)
            demand_processor.combine_demand_data(shelter_file="Generated data/Shelter_Demand.xlsx",pod_file="Generated data/POD_Demand.xlsx", network_processor=network_processor, output_file="Generated data/demand_data_dynamic.xlsx", debug_mode=True)
            demand_processor.generate_scenario_data()
            demand_processor.load_demand_data()

        else: 
            in_sample_data, scaled_demands = demand_processor.generate_in_sample_demand(args.num_sce, network_processor, debug_mode=args.debug)
            df_landfall_states = demand_processor.save_landfall_states(in_sample_data, network_processor, debug_mode=args.debug)
            shelter_demand_dict = demand_processor.allocate_demand(demand_data=scaled_demands, network_processor=network_processor, debug_mode=args.debug)
            pod_total_demand_dict = demand_processor.calculate_landfall_demand(landfall_df=df_landfall_states, pod_info_file="Data/PoD location information.xlsx", debug_mode=args.debug)
            pod_time_distributed_dict = demand_processor.distribute_demand_by_time(total_demand_dict=pod_total_demand_dict, network_processor=network_processor, debug_mode=args.debug)
            combined_demand_dict = demand_processor.combine_demand_data(shelter_data_dict=shelter_demand_dict, pod_data_dict=pod_time_distributed_dict, network_processor=network_processor, debug_mode=args.debug)
            demand_processor.populate_demand(combined_demand_dict)

        print("\nReading data...")  
        # Print the output
        print(f"All locations: {network_processor.V}")
        print(f"Warehouse/Supplier locations: {len(network_processor.V0)}, {network_processor.V0}")
        print(f"Shelter locations: {len(network_processor.V1)}, {network_processor.V1}")
        print(f"RSA locations: {len(network_processor.V2)}, {network_processor.V2}")
        print(f"PoD locations: {len(network_processor.V3)}, {network_processor.V3}")
        print(f"All resources: {network_processor.R}")
        print(f"Time periods: {network_processor.T}")
        print(f"Time Before Landfall: {network_processor.T_Bef}")
        print(f"Time After Landfall: {network_processor.T_Aft}")
        print(f"Scenario probabilities: {demand_processor.P}")

    # Run only the selected model
    if args.solution_method == 1:
        ext_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce)
        start_time_ext = time.time()

        ext_model, I_vars = ext_model_instance.ext_model(model_version=args.model)
        ext_model.optimize()

        end_time_ext = time.time()
        total_time_ext = end_time_ext - start_time_ext

        # Print and save results
        if ext_model.Status == gp.GRB.OPTIMAL:
            print(f"\nOptimal Objective Value: {ext_model.ObjVal}")
            output.ext_results(ext_model_instance, ext_model, args.instance_version, args.num_sce, end_time_ext - start_time_ext, network_processor.multiplier1, network_processor.multiplier2, network_processor.multiplier3, model_version=args.model, output_filename="Results/ext_results.xlsx")
            
            fixed_I_irt = ext_model_instance.extract_all_inventory(I_vars)
        else:
            print("\nModel did not solve to optimality.")

        print(f"Total Execution Time for Extended Model: {total_time_ext:.2f} seconds")

    if args.eval_oos:

        if 'fixed_I_irt' not in locals():
            raise RuntimeError("You must solve the model once, before running OOS evaluation.")        

        # -------- OOS Evaluation for Static 2SSP --------------------

        start_time = time.time()

        print("\nStarting OOS evaluation...")
        total_costs = []

        # Load inputs once
        demand_processor.load_input_files(population_file="Data/PoD location information.xlsx", distance_file="Data/Distance_podtoshelter.xlsx", network_processor=network_processor)

        sample_paths_file = "Data/real_OOS_paths.csv"
        
        # Load OOS sample paths (CSV)
        OS_paths_df = pd.read_csv(sample_paths_file)  # e.g., "real_sample_paths.csv"
        OS_paths = OS_paths_df.values

        # Compute landfall time once
        t_L = max(network_processor.T_Bef)

        for k in range(args.num_oos):
            print(f"\n🔁 OOS Scenario {k}")
            sample_path_row = OS_paths[k]
            s_t0 = eval(sample_path_row[0])  # current hurricane state at t = 0

            # Step 1: Clear previous demand
            demand_processor.d_its = {}

            # Generate pre-landfall demand at t=0
            demand_processor.generate_current_demand(t_bar=0, s_t=s_t0, network_processor=network_processor, data=data, solution_method=args.solution_method)

            # Clean the realized sample path row
            realized_path = [eval(state.replace('\x00', '')) for state in sample_path_row]
            # Make it shape (1, T) to act like [s, t]
            in_sample_paths = np.array([realized_path])  # Only s = 0

            # Generate pre-landfall demand from t=1 to t_L
            demand_processor.generate_future_demand(t_bar=0, s_t=s_t0, t_L=t_L, network_processor=network_processor, in_sample_paths=in_sample_paths, solution_method=args.solution_method)
            
            # Generate post-landfall demand
            demand_processor.generate_post_landfall_demand(t_L=t_L, in_sample_paths=in_sample_paths, network_processor=network_processor, solution_method=args.solution_method)

            # Step 3: Solve deterministic second-stage model with fixed I
            cost = ext_model_instance.OOS_2SSP(fixed_I_irt)
            total_costs.append(cost)
            print(f"   ➤ Cost = {cost:.2f}")

        total_eval_time = time.time() - start_time  # End timer
        print(f"\n Total OOS Evaluation Time: {total_eval_time:.2f} seconds")

        # Final summary
        avg_cost = np.mean(total_costs)
        print(f"\n✅ Average OOS Cost over {args.num_oos} scenarios: {avg_cost:.2f}")

        # Save to Excel
        output.oos_2SSP_results(costs=total_costs, instance_version=args.instance_version, num_oos=args.num_oos, total_time=total_eval_time, output_filename="Results/oos_2SSP_results.xlsx")

    elif args.solution_method == 2:
        benders_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce)
        benders_model_instance.master_problem(thetaLB, model_version=args.model)
        benders_model_instance.optimality_sub(model_version=args.model)
        benders_model_instance.feasibility_sub(model_version=args.model)
        start_time_benders = time.time()

        LB, UB, num_iter, num_Fcuts, num_Ocuts = benders_model_instance.benders(eps, thetaLB, args.time_limit, model_version=args.model)

        end_time_benders = time.time()
        total_time_benders = end_time_benders - start_time_benders

        output.benders_results(args.instance_version, args.num_sce, total_time_benders, LB, UB, model_version=args.model, output_filename="Results/benders_results.xlsx")
        print(f"Total Execution Time for Benders: {total_time_benders:.2f} seconds")

    elif args.solution_method == 3:
        demand_processor.compute_expected_demand()
        mvp_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce)
        mvp_model, _ = mvp_model_instance.ext_model(is_mvp=True)
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
        output.mvp_results(args.instance_version, args.num_sce, total_time_mvp, mvp_obj_value, network_processor.multiplier1, network_processor.multiplier2, network_processor.multiplier3, output_filename="Results/mvp_results.xlsx")

    elif args.solution_method == 4:
        rh_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce)
        
        # Provide the sample path file path here (update the path as needed)
        sample_paths_file = "Data/real_OOS_paths.csv"

        start_time_rh = time.time()
        avg_cost, total_costs = rh_model_instance.RH_evaluation(
            sample_paths=sample_paths_file,
            demand_processor=demand_processor,
            network_processor=network_processor, 
            start_index=args.start_index, # New
            num_paths=args.num_paths # New
        )
        end_time_rh = time.time()
        total_time_rh = end_time_rh - start_time_rh

        print(f"\nAverage RH Objective Value: {avg_cost:.2f}")
        print(f"Total RH Costs per Sample Path: {total_costs}")
        print(f"Total Execution Time for RH Evaluation: {total_time_rh:.2f} seconds")

        output.rh_results(
            instance_version=args.instance_version,
            num_sce=args.num_sce,
            total_time=total_time_rh,
            average_cost=avg_cost,
            path_costs=total_costs,
            output_filename="Results/rh_results.xlsx"
        )

    elif args.solution_method == 5:
        rh_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce)
        
        sample_paths_file = "Data/real_sample_paths.csv"  # make sure this exists on Palmetto

        start_time_rh = time.time()
        avg_cost, total_costs = rh_model_instance.RH_evaluation_parallel(
            sample_paths_file=sample_paths_file,
            demand_processor=demand_processor,
            network_processor=network_processor
        )
        end_time_rh = time.time()
        total_time_rh = end_time_rh - start_time_rh

        print(f"\nAverage RH Objective Value: {avg_cost:.2f}")
        print(f" Total RH Costs per Sample Path: {total_costs}")
        print(f" Total Execution Time for RH Evaluation: {total_time_rh:.2f} seconds")

        output.rh_results(
            instance_version=args.instance_version,
            num_sce=args.num_sce,
            total_time=total_time_rh,
            average_cost=avg_cost,
            path_costs=total_costs,
            output_filename="Results/rh_results_parallel.xlsx"
        )

    elif args.solution_method == 6:
        from multistage_model import MultiStageSPModel 
        from msp_helper import create_OSpaths

        # Initialize model instance
        sddp_model_instance = MultiStageSPModel(network_processor, demand_processor, params, args.travel_time)
        
        """
        # Path where we save or load OOS paths
        sample_paths_file = "Data/real_OOS_paths.csv"

        # === STEP 1: CREATE OOS PATHS (ONLY FIRST TIME) ===
        if not os.path.exists(sample_paths_file):
            print(f"[INFO] OOS paths file not found. Generating {demand_processor.nbOS} sample paths...")
            create_OSpaths(
                nbOS=demand_processor.nbOS,
                T_Bef=network_processor.T_Bef,
                k_init=sddp_model_instance.nodeLists[0][0],
                transition_dict=sddp_model_instance.transition_dict,
                output_file=sample_paths_file)
        else:
            print(f"[INFO] Using existing OOS paths file: {sample_paths_file}")
        exit(0)
        """
        # Run the SDDP algorithm
        print(f"\nSDDP Training in process ...")
        LB, train_time, total_iter = sddp_model_instance.solve_sddp()

        print(f"\nSDDP Training Complete")
        print(f"Iterations: {total_iter}")
        print(f"Training Time: {train_time:.2f} seconds")
        print(f"Final LB: {LB[-1]:.2f}")

        # Run OOS Evaluation for SDDP 
        print("\nStarting SDDP Out-of-Sample Evaluation...")
        sample_paths_file = "Data/real_OOS_paths.csv"

        results = sddp_model_instance.sddp_eval(sample_paths_file, demand_processor)

        mean_obj, CI, test_time = results

        print(f"Evaluation Time: {test_time:.2f} sec")


