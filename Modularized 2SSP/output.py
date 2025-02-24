import pandas as pd
import gurobipy as gp
import os

def store_objective_value(model, instance_version, num_sce, total_time, output_filename="solution_summary.xlsx"):
    """
    Stores the instance version, number of scenarios, and objective function value in an Excel file.
    If the file exists, it appends the new data; otherwise, it creates a new file.
    """
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
        print("⚠️ Model did not find an optimal solution. Objective not stored.")
