import pandas as pd
import copy

class DemandData:
    def __init__(self, num_sce, instance_version, nbOS):
        self.num_sce = num_sce
        self.nbOS = nbOS
        self.instance_version = instance_version
        self.d_its = {}  # Dictionary to store demand data
        self.full_demand_data = {}  # Store all data for later reuse

    def generate_scenario_data(self):
        self.sce = list(range(self.num_sce))  # Generate scenario indices
        self.P = [1 / self.num_sce] * self.num_sce  # Generate equal probabilities
        self.demand_file = f"demand_data_{self.instance_version}.xlsx"  # Demand file name
        self.sheet_names = [f"Scenario_{i}" for i in self.sce]  # Sheet names

    def load_demand_data(self):

        for s, sheet in enumerate(self.sheet_names):  # Scenario index starts from 0
            df = pd.read_excel(self.demand_file, sheet_name=sheet, header=0, index_col=0)  # Read demand data
            
            df = df.fillna(0)  # Replace NaN values with 0
            demand_array = df.values  # Convert DataFrame to NumPy array

            # Process demand data for each location and time period
            for location_index in range(demand_array.shape[0]):  # Rows = locations
                for time_period in range(demand_array.shape[1]):  # Columns = time periods
                    self.d_its[(location_index, time_period, s)] = demand_array[location_index, time_period]

            # Store full demand data
            # for loc in range(demand_array.shape[0]):
            #     for t in range(demand_array.shape[1]):
            #         self.full_demand_data[(loc, t, s)] = demand_array[loc, t]

    def update_d_its_for_roll(self, t_bar, roll_index):
 
        self.d_its.clear()  # Clear the previous data
        scenarios_per_roll = self.num_sce  # use same value unless you want different groups

        start_idx = roll_index * scenarios_per_roll
        end_idx = start_idx + scenarios_per_roll
        selected_scenarios = list(range(start_idx, min(end_idx, len(self.sheet_names))))

        self.sce = selected_scenarios
        self.P = [1 / len(selected_scenarios)] * len(selected_scenarios)

        for s in selected_scenarios:
            for (i, t, ss), val in self.full_demand_data.items():
                if ss == s and t >= t_bar:  # Only use time periods from t_bar onward
                    self.d_its[(i, t, s)] = val

    def compute_expected_demand(self):

        expected_demand = {}  # Dictionary to store averaged demand

        # Iterate over all locations and time periods
        for (i, t, s) in self.d_its:
            if (i, t) not in expected_demand:
                expected_demand[(i, t)] = 0  # Initialize sum
        
            expected_demand[(i, t)] += self.d_its[(i, t, s)] / self.num_sce

        self.d_its_expected = {(i, t, 0): expected_demand[(i, t)] for (i, t) in expected_demand}


