import pandas as pd
import gurobipy as gp
import os

def ext_results(model, instance_version, num_sce, total_time, output_filename="ext_results.xlsx"):

    if model.status == gp.GRB.OPTIMAL:
        obj_value = model.objVal  # Extract objective value

        # Create DataFrame with instance details and objective function
        df = pd.DataFrame({
            "Instance Version": [instance_version],
            "Number of Scenarios": [num_sce],
            "Objective Value": [obj_value],
            "Time": [total_time]
        })

        # Check if file exists
        if os.path.exists(output_filename):
            # Append to existing file
            with pd.ExcelWriter(output_filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, sheet_name="Objective", index=False, header=False, startrow=writer.sheets["Objective"].max_row)
        else:
            # Create a new file if it doesn’t exist
            with pd.ExcelWriter(output_filename, mode='w', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name="Objective", index=False)

        print(f"Objective value stored in {output_filename}")
    else:
        print("Model did not find an optimal solution. Objective not stored.")

def benders_results(instance_version, num_sce, LB, UB, output_filename="benders_results.xlsx"):

    # Create a DataFrame with Benders results
    df = pd.DataFrame({
        "Instance Version": [instance_version],
        "Number of Scenarios": [num_sce],
        "Lower Bound": [LB],
        "Upper Bound": [UB],
    })

    # Check if the file already exists
    if os.path.exists(output_filename):
        # Append to the existing file
        with pd.ExcelWriter(output_filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name="Benders_Results", index=False, header=False, startrow=writer.sheets["Benders_Results"].max_row)
    else:
        # Create a new file if it doesn’t exist
        with pd.ExcelWriter(output_filename, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Benders_Results", index=False)

    print(f"\nBenders results stored in {output_filename}")

def mvp_results(instance_version, num_sce, total_time_mvp, mvp_obj_value, output_filename="mvp_results.xlsx"):

    # Create a summary DataFrame
    df_summary = pd.DataFrame({
        "Instance Version": [instance_version],
        "Number of Scenarios": [num_sce],
        "MVP Objective Value": [mvp_obj_value],
        "MVP Execution Time (s)": [total_time_mvp]
    })

    # Save summary to an Excel file
    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\nMVP summary results stored in {output_filename}")

class TwoStageSPModel:
    def __init__(self, network_processor, demand_processor, num_sce):
        self.network_processor = network_processor  # Store network data
        self.demand_processor = demand_processor  # Store demand data
        self.num_sce = num_sce  # Store the number of scenarios

        # Initialize the Gurobi Model
        self.model = gp.Model("Two-Stage Stochastic Programming Model")