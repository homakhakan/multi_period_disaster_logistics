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
import copy
import json
import math

# === My own project modules ===
from demand_gen_indep import new_in_sample, demand_generation, inputs
from demand_data import DemandData
from network_data import NetworkData
from twostage_model import TwoStageSPModel
import output
from output import save_full_oos_demand


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_ins", type=str, required=True, help="MC instance folder name under mc_output, e.g. mc_H_segments_terminalstates")
    parser.add_argument("-ns", "--num_sce", type = int, help="Number of scenarios")
    parser.add_argument("-iv", "--instance_version", type=str, choices=["large", "small"])
    parser.add_argument("-c", "--commodity", type=str, choices=["water", "food", "both"], required=True, help="Commodity type: water, food, or both")
    parser.add_argument("-tl", "--time_limit", type=int, help="Time limit in seconds for running Benders")
    parser.add_argument("-no", "--nbOS", type=int, default=5, help="Number of sample paths for rolling horizon/sddp evaluation")
    parser.add_argument("-oos", "--num_oos", type=int, help="Number of out-of-sample scenarios for 2SSP evaluation")
    parser.add_argument("-eos", "--eval_oos", action="store_true", help="Run out-of-sample evaluation for 2SSP after solving extended model")
    parser.add_argument("-sm", "--solution_method", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], required=True, help="Solution method: 1. extended, 2. benders, 3. mvp, 4.RH, 5.RH(parallel), 6.multi-stage (SDDP), 7. Perfect Info, 8.Wait-and-See")
    parser.add_argument("-m", "--model", type=int, choices=[1, 2], required=True, help="Model version: 1 = Using fixed arc capacity, 2 = with vehicle routing decisions")
    parser.add_argument("-tt", "--travel_time", type=int, choices=[1, 2], help="Travel-time mode: 1. All arcs take 1 period to travel, 2. Some arcs might take more than 1 period to travel")
    parser.add_argument("-ls", "--last_stage", type=int, choices=[1, 2], help="Last stage mode: 1. Last stage includes one single period, 2. Last stage includes multiple periods")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-tilda", "--t_tilda", type=int, help="Waiting time (t_tilda) before starting the Wait-and-See (WS) 2SSP model") 
    parser.add_argument("-nsm", "--num_sce_mean", type=int, default=None, help="Number of scenarios used to compute mean demand for deterministic RH (only used when -sm 4 and -ns 1)") # Deterministic mean value RH

    args = parser.parse_args()

    with open("fixed_params.yaml", "r") as f:
        params = yaml.safe_load(f)

    thetaLB = float(params["THETA_LB"])
    eps = float(params["EPS"])
    network_processor = NetworkData(args.instance_version, params) # Create an instance of the NetworkData class
    network_processor.commodity = args.commodity
    network_processor.define_sets() # Now call the methods using this instance
    network_processor.define_parameters(args.travel_time)

    demand_processor = DemandData(args.num_sce, args.instance_version, args.nbOS, params) # Create an instance of the DemandData class

    # ---------- Simulated In-Sample Demand Setup for static 2SSP-----------
    # Create different scenarios for solving static 2SSP. Based on specifying -ns (number of scenarios), we can have that many scenarios created in "demand_data_dynamic"
    # print("\nGenerating in-sample demand...")
    if args.instance_version == "small":
        dist_file = "Data/Distance_podtoshelter_small.xlsx"
    elif args.instance_version == "large":
        dist_file = "Data/Distance_podtoshelter_large.xlsx"

    demand_processor.load_input_files(population_file="Data/PoD location information.xlsx", distance_file=dist_file, network_processor=network_processor)
    if args.solution_method == 4:
        demand_processor.P = [1 / args.num_sce] * args.num_sce
        demand_processor.sce = list(range(args.num_sce))
        
    if args.solution_method not in [4, 6]:
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
    print(f"All locations: {network_processor.V}")
    print(f"Warehouse/Supplier locations: {len(network_processor.V0)}, {network_processor.V0}")
    print(f"Shelter locations: {len(network_processor.V1)}, {network_processor.V1}")
    print(f"RSA locations: {len(network_processor.V2)}, {network_processor.V2}")
    print(f"PoD locations: {len(network_processor.V3)}, {network_processor.V3}")
    print(f"All resources: {network_processor.R}")
    print(f"Time periods: {network_processor.T}")
    print(f"Time Before Landfall: {network_processor.T_Bef}")
    print(f"Time After Landfall: {network_processor.T_Aft}")
    # print(f"Scenario probabilities: {demand_processor.P}")

    # Run only the selected model
    if args.solution_method == 1:
        ext_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce, args.num_sce_mean)
        start_time_ext = time.time()

        ext_model, I_vars = ext_model_instance.ext_model(model_version=args.model)
        ext_model.optimize()

        end_time_ext = time.time()
        total_time_ext = end_time_ext - start_time_ext

        # Print and save results
        if ext_model.Status == gp.GRB.OPTIMAL:
            print(f"\nOptimal Objective Value: {ext_model.ObjVal}")
            output.ext_results(ext_model_instance, ext_model, args.instance_version, args.num_sce, end_time_ext - start_time_ext, network_processor.multiplier_penalty_cost, network_processor.multiplier_init_inv, network_processor.multiplier_arc, model_version=args.model, output_filename="Results/ext_results.xlsx")
            
            fixed_I_irt = ext_model_instance.extract_all_inventory(I_vars)
        else:
            print("\nModel did not solve to optimality.")

        print(f"Total Execution Time for Extended Model: {total_time_ext:.2f} seconds")

    elif args.solution_method == 2:
        benders_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce, args.num_sce_mean)
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
        print("Solving Mean Value Problem (MVP)...")

        # print("Original stochastic demand (demand_processor.d_its):")
        # for key in sorted(demand_processor.d_its):
        #     val = demand_processor.d_its[key]
        #     if abs(val) >= 1e-6:
        #         print(f"d_its{key} = {val:.2f}")

        # Make a deep copy of the original demand processor to retain stochastic demand
        copy_d_its = copy.deepcopy(demand_processor.d_its)

        # print("Copy of stochastic demand (copy_d_its):")
        # for key in sorted(copy_d_its):
        #     val = copy_d_its[key]
        #     if abs(val) >= 1e-6:
        #         print(f"d_its{key} = {val:.2f}")
        
        # Verify the deep copy matches the original
        for key in demand_processor.d_its:
            orig_val = demand_processor.d_its[key]
            copy_val = copy_d_its[key]

            # If key missing or values differ more than tolerance, throw error
            if abs(orig_val - copy_val) > 1e-6:
                raise ValueError(f"❌ Demand mismatch found at key {key}: original={orig_val}, copy={copy_val}")

        print("✅ Deep copy check passed: copy_d_its matches original demand exactly.")
        
        # Initialize an empty dictionary to store the mean demand across scenarios
        mean_demand = {}
        # Loop through all (i, t, s) keys and accumulate the average demand across scenarios
        for (i, t, s), val in demand_processor.d_its.items():
            key = (i, t)
            if key not in mean_demand:
                mean_demand[key] = 0.0
            mean_demand[key] += val / demand_processor.num_sce
        # Convert the mean demand into a deterministic demand dictionary. Assign scenario index 0 to each (i, t) pair
        det_demand = {(i, t, 0): val for (i, t), val in mean_demand.items()}

        # Replace demand and number of scenarios with deterministic mean value demand
        demand_processor.d_its = det_demand
        demand_processor.num_sce = 1
        demand_processor.P = [1.0]  # Set scenario probability to 1 since we only have one

        # # Optional print for debugging
        # print(f"Number of scenarios: {demand_processor.num_sce}")
        # print(f"Scenario probabilities: {demand_processor.P}")
        # for key in sorted(demand_processor.d_its):
        #     print(f"  {key}: {demand_processor.d_its[key]}")

        model_instance = TwoStageSPModel(network_processor, demand_processor, 1, args.num_sce_mean)
        model, _ = model_instance.ext_model(model_version=args.model)
        model.optimize()

        # Extract first-stage solution
        if model.status == gp.GRB.OPTIMAL:
            I_mvp_values = {
                (i, r, t): model.getVarByName(f"I_{i}_{r}_{t}").X
                for i in range(model_instance.network_processor.V)
                for r in range(model_instance.network_processor.R)
                for t in range(model_instance.network_processor.T)}
        else:
            print(f"MVP model did NOT solve to optimality. Status code = {model.status}")
            exit(0)

        # Restore original demand for evaluation
        model_instance.demand_processor.d_its = copy_d_its
        model_instance.demand_processor.num_sce = args.num_sce
        model_instance.demand_processor.P = [1 / args.num_sce] * args.num_sce

        # print("Original stochastic demand after restoring (model_instance.demand_processor.d_its):")
        # for key in sorted(model_instance.demand_processor.d_its):
        #     val = model_instance.demand_processor.d_its[key]
        #     if abs(val) >= 1e-6:
        #         print(f"d_its{key} = {val:.2f}")
        
        model_instance.optimality_sub()
        mvp_obj_value = model_instance.compute_mvp_objective(I_mvp_values)
        print(f"True MVP Objective Value, averaged over all scenarios: {mvp_obj_value}")

    elif args.solution_method == 4:
        rh_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce, args.num_sce_mean)
        
        # Provide the sample path file path here (update the path as needed)
        instance_dir = os.path.join("mc_output", args.mc_ins)
        suffix = args.mc_ins.replace("mc_", "")  # "60_2_5"
        sample_paths_file = os.path.join(instance_dir, f"real_OOS_paths_{suffix}.csv")

        avg_cost, total_costs, total_time = rh_model_instance.RH_evaluation(
            sample_paths=sample_paths_file,
            demand_processor=demand_processor,
            network_processor=network_processor,
        )

        output.rh_results(
            instance_version=args.instance_version,
            num_sce=args.num_sce,
            average_cost=avg_cost,
            path_costs=total_costs,
            total_time=total_time,
            output_filename="Results/rh_results.xlsx"
        )

    elif args.solution_method == 5:
        rh_model_instance = TwoStageSPModel(network_processor, demand_processor, args.num_sce, args.num_sce_mean)
        
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
        # import cProfile
        # import pstats
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
        from multistage_model import MultiStageSPModel 
        from msp_helper import create_OSpaths

        # Initialize model instance
        sddp_model_instance = MultiStageSPModel(network_processor, demand_processor, params, args.travel_time, args.last_stage, args.mc_ins)
        
        '''
        # Load MC parameters used for naming
        instance_dir = os.path.join("mc_output", args.mc_ins) # Instance-specific MC folder (required)
        with open("mc_data/inputs.json", "r") as f:
            raw = json.load(f)
        mc_args = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}
        H = mc_args["H"]
        segments = mc_args["segments"]
        n_states_at_terminal = mc_args["n_states_at_terminal"]

        # (Optional safety check)
        expected = f"mc_{H}_{segments}_{n_states_at_terminal}"
        if args.mc_ins != expected:
            raise ValueError(f"--mc_ins={args.mc_ins} but inputs.json implies {expected}")
        
        # Path where we save or load OOS paths
        sample_paths_file = os.path.join(instance_dir, f"real_OOS_paths_{H}_{segments}_{n_states_at_terminal}.csv")

        # CREATE OOS PATHS (ONLY ONE TIME)
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
        '''
        
        # # === PROFILE SDDP TRAINING ===
        # print(f"\n[PROFILING] SDDP Training in process ...")
        # prof_train = cProfile.Profile()
        # prof_train.enable()
        
        # Run the SDDP algorithm
        print(f"\nSDDP Training in process ...")
        LB, train_time, total_iter = sddp_model_instance.solve_sddp()

        # prof_train.disable()
        # prof_train.dump_stats("profile_sddp_train.prof")

        print(f"\nSDDP Training Complete")
        print(f"Iterations: {total_iter}")
        print(f"Training Time: {train_time:.2f} seconds")
        print(f"Final LB: {LB[-1]:.2f}")

        # Run OOS Evaluation for SDDP 
        print("\nStarting SDDP Out-of-Sample Evaluation...")
        instance_dir = os.path.join("mc_output", args.mc_ins)
        suffix = args.mc_ins.replace("mc_", "")  # "60_2_5"
        sample_paths_file = os.path.join(instance_dir, f"real_OOS_paths_{suffix}.csv")

        results = sddp_model_instance.sddp_eval(sample_paths_file, demand_processor)

        mean_obj, CI, test_time = results

        print(f"Evaluation Time: {test_time:.2f} sec")

    elif args.solution_method == 7:
        print("▶ Computing Perfect Information solution...")

        num_sce = args.num_sce  # Number of scenarios previously generated
        # print("number of scenarios:", num_sce)
        scenario_costs = []

        # # Sample print to check demand data
        # print("Sample from demand_processor.d_its:")
        # for idx, key in enumerate(demand_processor.d_its):
        #     print(f"{key}: {demand_processor.d_its[key]}")

        copy_d_its = copy.deepcopy(demand_processor.d_its)

        for s in range(num_sce):
            # print(f"→ Solving deterministic model with perfect information for scenario {s}...")

            # Create deterministic demand dict with scenario index 0
            det_demand = {(i, t, 0): copy_d_its[(i, t, s)] for (i, t, ss) in copy_d_its if ss == s}

            # # Print entries of det_demand
            # print(f"▶ Deterministic demand for scenario {s}:")
            # for idx, key in enumerate(det_demand):
            #     print(f"    {key}: {det_demand[key]}")

            # Replace demand and number of scenarios
            demand_processor.d_its = det_demand
            demand_processor.num_sce = 1
            demand_processor.P = [1.0]

            # # Sample print to check demand data
            # print("Sample from demand_processor.d_its:")
            # for idx, key in enumerate(demand_processor.d_its):
            #     print(f"{key}: {demand_processor.d_its[key]}")

            model_instance = TwoStageSPModel(network_processor, demand_processor, 1, args.num_sce_mean)
            model, _ = model_instance.ext_model(model_version=args.model)
            model.optimize()

            # # Optional debug prints
            # print(f"→ demand_processor.num_sce: {demand_processor.num_sce}")
            # print("▶ Scenario probabilities inside model:", demand_processor.P)

            if model.status == gp.GRB.OPTIMAL:
                scenario_cost = model.ObjVal
                print(f"Scenario {s} cost: {scenario_cost:.2f}")
                scenario_costs.append(scenario_cost)
            else:
                print(f"Model for scenario {s} did not solve to optimality.")
                exit(0)

        avg_cost = sum(scenario_costs) / len(scenario_costs)
        print(f"\n✅ Average Perfect Information Cost: {avg_cost:.2f}")

    elif args.solution_method == 8:

        t_tilda = args.t_tilda
        print(f"Running Wait-and-See 2SSP with t_tilda = {t_tilda}")

        Wait_and_See = TwoStageSPModel(network_processor, demand_processor, args.num_sce, args.num_sce_mean)
        ws_model, I_vars = Wait_and_See.WS_model(t_tilda)
        ws_model.optimize()

        if ws_model.Status == gp.GRB.OPTIMAL:
            print(f"\nObjective value (Wait-and-See, t_tilda={t_tilda}): {ws_model.objVal}")
            fixed_I_irt = Wait_and_See.extract_all_inventory(I_vars, t_tilda)
        else:
            print("\nModel did not solve to optimality.")

    if args.eval_oos:

        if 'fixed_I_irt' not in locals():
            raise RuntimeError("You must solve the model once, before running OOS evaluation.")        

        # -------- OOS Evaluation for Static 2SSP --------------------

        start_time = time.time()

        print("\nStarting OOS evaluation...")
        total_costs = []

        # Load inputs once
        if args.instance_version == "small":
            dist_file = "Data/Distance_podtoshelter_small.xlsx"
        elif args.instance_version == "large":
            dist_file = "Data/Distance_podtoshelter_large.xlsx"
        demand_processor.load_input_files(population_file="Data/PoD location information.xlsx", distance_file=dist_file, network_processor=network_processor)

        instance_dir = os.path.join("mc_output", args.mc_ins)
        suffix = args.mc_ins.replace("mc_", "")  # "60_2_5"
        sample_paths_file = os.path.join(instance_dir, f"real_OOS_paths_{suffix}.csv")
        
        # Load OOS sample paths (CSV)
        OS_paths_df = pd.read_csv(sample_paths_file)  # e.g., "real_sample_paths.csv"
        OS_paths = OS_paths_df.values

        # Compute landfall time once
        t_L = max(network_processor.T_Bef)

        for k in range(args.num_oos):
            # print(f"\n OOS Scenario {k}")
            sample_path_row = OS_paths[k]
            s_t0 = eval(sample_path_row[0])  # current hurricane state at t = 0

            # Step 1: Clear previous demand
            demand_processor.d_its = {}

            # Generate pre-landfall demand at t=0
            demand_processor.generate_current_demand(t_bar=0, s_t=s_t0, network_processor=network_processor, solution_method=args.solution_method)

            # Clean the realized sample path row
            realized_path = [eval(state.replace('\x00', '')) for state in sample_path_row]
            # Make it shape (1, T) to act like [s, t]
            in_sample_paths = np.array([realized_path])  # Only s = 0

            # Generate pre-landfall demand from t=1 to t_L
            demand_processor.generate_future_demand(t_bar=0, t_L=t_L, network_processor=network_processor, in_sample_paths=in_sample_paths, solution_method=args.solution_method)
            
            # Generate post-landfall demand
            demand_processor.generate_post_landfall_demand(t_L=t_L, in_sample_paths=in_sample_paths, network_processor=network_processor, solution_method=args.solution_method)

            # Step 3: Solve deterministic second-stage model with fixed I
            if args.solution_method == 1:
                cost = ext_model_instance.OOS_2SSP(fixed_I_irt, args)
            elif args.solution_method == 8:
                cost = Wait_and_See.OOS_2SSP(fixed_I_irt, args)
            else:
                raise ValueError(f"OOS evaluation not implemented for solution method {args.solution_method}")
            
            total_costs.append(cost)
            # print(f"   ➤ Cost = {cost:.2f}")

        total_eval_time = time.time() - start_time  # End timer
        print(f"\nTotal OOS Evaluation Time: {total_eval_time:.2f} seconds")

        # Final summary
        avg_cost = np.mean(total_costs)
        std_obj = np.std(total_costs)   # sample standard deviation
        nbOS = len(total_costs)                 # number of OOS scenarios

        ci_low = avg_cost - 1.96 * std_obj / np.sqrt(nbOS)
        ci_high = avg_cost + 1.96 * std_obj / np.sqrt(nbOS)
        CI = avg_cost - ci_low

        print(f"\nTwo-Stage OOS Evaluation Results")
        print(f"μ ± 1.96*σ/√NS = {avg_cost:.2f} ± {CI:.2f}")
        print(f"95% Confidence Interval = [{ci_low:.2f}, {ci_high:.2f}]")

        # print(f"\n Average OOS Cost over {args.num_oos} scenarios: {avg_cost:.2f}")

        # Save to Excel
        output.oos_2SSP_results(costs=total_costs, instance_version=args.instance_version, num_oos=args.num_oos, total_time=total_eval_time, output_filename="Results/oos_2SSP_results.xlsx")




        
