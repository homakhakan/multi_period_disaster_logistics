import pandas as pd
import gurobipy as gp
import os
from openpyxl import load_workbook

def ext_results(model_instance, model, instance_version, num_sce, total_time, multiplier1, multiplier2, multiplier3, model_version, output_filename="ext_results.xlsx"):

    if model.status == gp.GRB.OPTIMAL:
        obj_value = model.objVal  # Extract objective value

        # Extract variables and sets from model_instance
        P = model_instance.demand_processor.P
        p_i = model_instance.network_processor.p_i
        cost_ir = model_instance.network_processor.cost_ir
        m_ij = model_instance.network_processor.m_ij
        V = model_instance.network_processor.V
        V0 = model_instance.network_processor.V0
        R = model_instance.network_processor.R
        T_Bef = model_instance.network_processor.T_Bef
        T_Aft = model_instance.network_processor.T_Aft
        T = model_instance.network_processor.T
        A = model_instance.network_processor.A
        sce = range(len(P))

        # Extract variables and their values
        U = {var.VarName: var.X for var in model.getVars() if var.VarName.startswith("U_")}
        f = {var.VarName: var.X for var in model.getVars() if var.VarName.startswith("f_")}

        # Initialize cost components
        penalty_cost_before = 0
        penalty_cost_after = 0
        transport_cost_before = 0
        transport_cost_after = 0

        # Loop over scenarios
        for s in sce:
            for t in T_Bef:
                for i in range(V):
                    penalty_cost_before += P[s] * p_i[i] * U[f"U_{i}_{t}_{s}"]

                for r in range(R):
                    for i in V0:
                        for j in range(V):
                            if (i, j) in A:
                                key = f"f_{i}_{j}_{r}_{t}_{s}"
                                transport_cost_before += P[s] * cost_ir[i][r] * f.get(key, 0)

                    for i in range(V):
                        for j in range(V):
                            if (i, j) in A:
                                key = f"f_{i}_{j}_{r}_{t}_{s}"
                                transport_cost_before += P[s] * m_ij[i][j] * f.get(key, 0)

            for t in T_Aft:
                for i in range(V):
                    penalty_cost_after += P[s] * p_i[i] * U[f"U_{i}_{t}_{s}"]

                for r in range(R):
                    for i in V0:
                        for j in range(V):
                            if (i, j) in A:
                                key = f"f_{i}_{j}_{r}_{t}_{s}"
                                transport_cost_after += P[s] * cost_ir[i][r] * f.get(key, 0)

                    for i in range(V):
                        for j in range(V):
                            if (i, j) in A:
                                key = f"f_{i}_{j}_{r}_{t}_{s}"
                                transport_cost_after += P[s] * m_ij[i][j] * f.get(key, 0)

        # Compute totals
        total_before = penalty_cost_before + transport_cost_before
        total_after = penalty_cost_after + transport_cost_after

        # Create DataFrame with instance details and objective function
        df = pd.DataFrame({
            "Instance Version": [instance_version],
            "Model Version": [model_version],
            "Number of Scenarios": [num_sce],
            "Objective Value": [obj_value],
            "Time (seconds)": [total_time],
            "Total Cost (Before Landfall)": [total_before],
            "Total Cost (After Landfall)": [total_after],
            "Penalty Cost (Before Landfall)": [penalty_cost_before],
            "Penalty Cost (After Landfall)": [penalty_cost_after],
            "Transportation Cost (Before Landfall)": [transport_cost_before],
            "Transportation Cost (After Landfall)": [transport_cost_after],
            "Penalty Multiplier": [multiplier1],
            "Travel Cost Multiplier": [multiplier2],
            "Arc Capacity Multiplier": [multiplier3]
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

        print(f"Extended results stored in {output_filename}")
    else:
        print("Model did not find an optimal solution. Objective not stored.")

def oos_2SSP_results(costs, instance_version, num_oos, output_filename="oos_results.xlsx", total_time=None):
    """
    Save individual and average out-of-sample (OOS) costs to an Excel file.

    """

    df = pd.DataFrame({
        "Scenario": list(range(num_oos)),
        "Cost": costs
    })

    df_avg = pd.DataFrame({
        "Scenario": ["Average"],
        "Cost": [sum(costs) / len(costs)],
        "Time (sec)": [total_time if total_time is not None else None]
    })

    combined_df = pd.concat([df, df_avg], ignore_index=True)

    sheet_name = "OOS_Results"

    # Save to Excel
    if os.path.exists(output_filename):
        with pd.ExcelWriter(output_filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=writer.sheets[sheet_name].max_row)
    else:
        with pd.ExcelWriter(output_filename, mode='w', engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nOOS results stored in {output_filename}")


import pandas as pd

def save_full_oos_demand(full_demand, network_processor, output_file="full_oos_demand.xlsx"):
    """
    Save the full OOS demand dictionary to Excel with locations as rows and time periods as columns.
    
    Args:
        full_demand (dict): Dictionary {(i, t): demand}
        network_processor: Instance of NetworkData to access V and T
        output_file (str): Path to the output Excel file
    """
    V = network_processor.V
    T = network_processor.T

    # Create a DataFrame with index=1...V and columns=0...T-1
    df = pd.DataFrame(index=range(1, V+1), columns=range(T))

    for (i, t), val in full_demand.items():
        df.at[i + 1, t] = round(val, 3)  # +1 to match Excel indexing (starts from 1)

    df.fillna(0, inplace=True)

    # Add 'Index' column like your screenshot
    df.insert(0, "Index", df.index)

    df.to_excel(output_file, index=False)
    print(f"✅ Full OOS demand saved to: {output_file}")

def benders_results(instance_version, num_sce, total_time_benders, LB, UB, model_version, output_filename="benders_results.xlsx"):

    # Create a DataFrame with Benders results
    df = pd.DataFrame({
        "Instance Version": [instance_version],
        "Model Version": [model_version],
        "Number of Scenarios": [num_sce],
        "Lower Bound": [LB],
        "Upper Bound": [UB],
        "Time (seconds)": [total_time_benders]
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

def mvp_results(instance_version, num_sce, total_time_mvp, mvp_obj_value, multiplier1, multiplier2, multiplier3, output_filename="mvp_results.xlsx"):
    # Create a DataFrame with MVP results
    df = pd.DataFrame({
        "Instance Version": [instance_version],
        "Number of Scenarios": [num_sce],
        "Objective Value": [mvp_obj_value],
        "Time (seconds)": [total_time_mvp],
        "Penalty Multiplier": [multiplier1],
        "Travel Cost Multiplier": [multiplier2],
        "Arc Capacity Multiplier": [multiplier3]
    })

    # Check if the file already exists
    if os.path.exists(output_filename):
        with pd.ExcelWriter(output_filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name="MVP_Results", index=False, header=False, startrow=writer.sheets["MVP_Results"].max_row)
    else:
        # Create a new file and write data with headers
        with pd.ExcelWriter(output_filename, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="MVP_Results", index=False)

    print(f"\nMVP results stored in {output_filename}")

def rh_results(instance_version, num_sce, total_time, average_cost, path_costs, output_filename="rh_results.xlsx"):

    # Summary DataFrame (overall average)
    summary_df = pd.DataFrame({
        "Instance Version": [instance_version],
        "Number of Scenarios": [num_sce],
        "Average Objective Value": [average_cost],
        "Time (seconds)": [total_time]
    })

    # Path-wise cost DataFrame
    path_df = pd.DataFrame({"Sample Path Index": list(range(len(path_costs))),
                             "Objective Value": path_costs})

    if os.path.exists(output_filename):
        with pd.ExcelWriter(output_filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            summary_df.to_excel(writer, sheet_name="RH_Summary", index=False, header=False, startrow=writer.sheets["RH_Summary"].max_row)
            path_df.to_excel(writer, sheet_name="RH_Path_Costs", index=False, header=False, startrow=writer.sheets["RH_Path_Costs"].max_row)
    else:
        with pd.ExcelWriter(output_filename, mode='w', engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name="RH_Summary", index=False)
            path_df.to_excel(writer, sheet_name="RH_Path_Costs", index=False)

    print(f"\nRolling Horizon results stored in {output_filename}")