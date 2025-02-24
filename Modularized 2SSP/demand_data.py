import pandas as pd

class DemandData:
    def __init__(self, num_sce, instance_version):
        self.num_sce = num_sce
        self.instance_version = instance_version

    def generate_scenario_data(self):
        """
        Generate scenario data based on instance version and number of scenarios.
        """
        sce = list(range(self.num_sce))  # Generate scenario indices
        P = [1 / self.num_sce] * self.num_sce  # Generate equal probabilities
        demand_file = f"demand_data_{self.instance_version}.xlsx"  # Demand file name
        sheet_names = [f"Scenario_{i}" for i in sce]  # Sheet names

        print(f"✅ Number of scenarios: {self.num_sce}")
        print(f"✅ Generated sheet names: {sheet_names}")

        return {"sce": sce, "P": P, "demand_file": demand_file, "sheet_names": sheet_names}

    def load_demand_data(self, demand_file, sheet_names):
        """
        Load demand data from the Excel file for all scenarios.
        """
        d_its = {}  # Dictionary to store demand data

        for s, sheet in enumerate(sheet_names):  # Scenario index starts from 0
            df = pd.read_excel(demand_file, sheet_name=sheet, header=0, index_col=0)  # Read demand data
            
            df = df.fillna(0)  # Replace NaN values with 0
            demand_array = df.values  # Convert DataFrame to NumPy array

            # Process demand data for each location and time period
            for location_index in range(demand_array.shape[0]):  # Rows = locations
                for time_period in range(demand_array.shape[1]):  # Columns = time periods
                    d_its[(location_index, time_period, s)] = demand_array[location_index, time_period]

        return d_its  # Return processed demand data dictionary
