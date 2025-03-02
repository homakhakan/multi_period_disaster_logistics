import pandas as pd

class DemandData:
    def __init__(self, num_sce, instance_version):
        self.num_sce = num_sce
        self.instance_version = instance_version

    def generate_scenario_data(self):
        self.sce = list(range(self.num_sce))  # Generate scenario indices
        self.P = [1 / self.num_sce] * self.num_sce  # Generate equal probabilities
        self.demand_file = f"demand_data_{self.instance_version}.xlsx"  # Demand file name
        self.sheet_names = [f"Scenario_{i}" for i in self.sce]  # Sheet names

    def load_demand_data(self):

        self.d_its = {}  # Dictionary to store demand data

        for s, sheet in enumerate(self.sheet_names):  # Scenario index starts from 0
            df = pd.read_excel(self.demand_file, sheet_name=sheet, header=0, index_col=0)  # Read demand data
            
            df = df.fillna(0)  # Replace NaN values with 0
            demand_array = df.values  # Convert DataFrame to NumPy array

            # Process demand data for each location and time period
            for location_index in range(demand_array.shape[0]):  # Rows = locations
                for time_period in range(demand_array.shape[1]):  # Columns = time periods
                    self.d_its[(location_index, time_period, s)] = demand_array[location_index, time_period]
    
    def calculate_expected_demand(self, output_file):

        input_file = f"demand_data_{self.instance_version}.xlsx"  # Use the instance's demand file
        num_scenarios = self.num_sce  # Use the number of scenarios from the instance

        # Read data from each scenario sheet
        scenario_data = pd.read_excel(input_file, sheet_name=None)

        # Initialize a DataFrame to store the cumulative sum of demands
        expected_demand = pd.DataFrame(0.0, index=range(43), columns=range(12))

        # Iterate over all scenarios and sum the demands
        for scenario_num in range(num_scenarios):
            scenario_sheet = f'Scenario_{scenario_num}'
            if scenario_sheet in scenario_data:
                scenario_df = scenario_data[scenario_sheet]
                # Skip the 'Index' column and add the values to the cumulative sum
                expected_demand += scenario_df.iloc[:, 1:].values

        # Calculate the expected demand by dividing by the number of scenarios
        expected_demand /= num_scenarios

        # Add the 'Index' column back
        expected_demand.insert(0, 'Index', range(1, 44))

        # Save the expected demand to a new sheet in the output file
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            expected_demand.to_excel(writer, sheet_name='Expected_Demand', index=False)

        print(f"Expected demand saved to {output_file}")

    def load_expected_demand(self, expected_demand_file, sheet_name="Expected_Demand"):

        d_it = {}  # Dictionary to store demand data in (location, time) format

        # Read the expected demand sheet, keeping the index column
        df = pd.read_excel(expected_demand_file, sheet_name=sheet_name, header=0)

        # Replace NaN values with 0 (if applicable)
        df = df.fillna(0)

        # Extract the index column to get location IDs
        location_indices = df['Index'].values

        # Remove the 'Index' column from the DataFrame
        df = df.drop(columns='Index')

        # Convert DataFrame to a NumPy array for iteration
        demand_array = df.values

        # Process demand data for each location and time period using the location IDs
        for row_index, location_id in enumerate(location_indices):
            for time_period in range(demand_array.shape[1]):  # Columns represent time periods
                d_it[(location_id - 1, time_period)] = demand_array[row_index, time_period]  # Adjust by subtracting 1

        return d_it


    

