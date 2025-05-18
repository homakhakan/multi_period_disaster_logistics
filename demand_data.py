import pandas as pd
import numpy as np
import copy
import os

class DemandData:
    def __init__(self, num_sce, instance_version, nbOS, params):
        self.num_sce = num_sce
        self.nbOS = nbOS
        self.instance_version = instance_version
        self.d_its = {}  # Dictionary to store demand data
        self.full_demand_data = {}  # Store all data for later reuse
        self.population = None # PoD-level population, loaded once
        self.weights = None # Inverse distance weights (PoD → Shelter)

        self.params = params
        self.std_pod = float(params["std_pod"])

    def generate_scenario_data(self):
        self.sce = list(range(self.num_sce))  # Generate scenario indices
        self.P = [1 / self.num_sce] * self.num_sce  # Generate equal probabilities
        self.demand_file = "Generated data/demand_data_dynamic.xlsx"  # Demand file name
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

    def load_oos_inputs(self, population_file, distance_file, network_processor):
        """
        Loads PoD population and inverse-distance weights from PoDs to shelters.
        Stores results in self.population and self.weights.
        """
        # Load PoD population
        pod_info_df = pd.read_excel(population_file, sheet_name="Sheet1")
        self.pod_info_df = pod_info_df
        self.population = pod_info_df["Pop. of County"].to_numpy()

        # Load distance matrix (PoDs to shelters)
        full_matrix = pd.read_excel(distance_file, header=None)
        num_pods = len(network_processor.V3)
        num_shelters = len(network_processor.V1)

        distance_matrix = full_matrix.iloc[:num_pods, :num_shelters]

        # Compute inverse distance weights
        inverse_distances = 1 / distance_matrix.replace(0, float("inf"))
        self.weights = inverse_distances.div(inverse_distances.sum(axis=1), axis=0)

    def pre_landfall_demand_S_2SSP(self, sample_path_row, network_processor, demand_processor, t_L = None):
        """
        Generates full deterministic demand (from t=0 to t=t_L) for a given sample path row.
        Updates demand_processor.d_its with (i, t, 0) keys.
        
        """
        from demand_generation import demand_deterministic_landfall, data
        
        for t in range(t_L + 1):
            s_t = eval(sample_path_row[t])
            track_err, intensity = s_t

            # Step 1: Compute PoD demand
            demand_vector = demand_deterministic_landfall(data, track_err, int(intensity), t)
            scaled_pod_demand = demand_processor.population * np.array(demand_vector)

            # Step 2: Allocate to shelters
            demand_df = pd.DataFrame({t: scaled_pod_demand})
            shelter_demands = demand_processor.weights.multiply(demand_df[t], axis=0).sum(axis=0).values

            # Step 3: Store in d_its with scenario index 0
            for shelter_idx, i in enumerate(network_processor.V1):
                demand_processor.d_its[(i, t, 0)] = shelter_demands[shelter_idx]

    def post_landfall_demand_S_2SSP(self, sample_path_row, network_processor, demand_processor, t_L = None):
        """
        Generate deterministic PoD-level demand for t in T_Aft based on hurricane category at landfall
        (i.e., state at t_L) from one realized sample path.

        Updates demand_processor.d_its[(i, t, 0)] with demand for PoD locations (V3).
        """
        # Get landfall state (at t_L)
        track_err_L, hurr_cat_L = eval(sample_path_row[t_L])
        cat = int(hurr_cat_L)

        if cat == 0 or pd.isna(cat):
            # Assign 0 demand to all PoD locations after landfall
            for i in network_processor.V3:
                for t in network_processor.T_Aft:
                    demand_processor.d_its[(i, t, 0)] = 0.0
        else:
            # Load total demand per PoD location based on hurricane category
            demand_col = f"I*p_c (cat {cat})"
            evac_col = "% 1- (Likely to evacuate)"
            total_demand = demand_processor.pod_info_df[demand_col] * demand_processor.pod_info_df[evac_col]
            total_demand = total_demand.to_numpy()  # array of shape (num_pods,)

            # Generate time-distribution weights (normal shape)
            mean = np.mean(network_processor.T_Aft)
            std = demand_processor.std_pod
            weights = np.array([
                np.exp(-((t - mean) ** 2) / (2 * std ** 2)) for t in network_processor.T_Aft
            ])
            weights /= weights.sum()

            # Assign time-distributed demand to each PoD
            for idx, i in enumerate(network_processor.V3):
                for w_idx, t in enumerate(network_processor.T_Aft):
                    demand_processor.d_its[(i, t, 0)] = total_demand[idx] * weights[w_idx]

    def rh_generate_current_demand(self, t_bar, s_t, network_processor, data):
        """
        Generate deterministic shelter-level demand at time t_bar based on the realized hurricane state s_t.
        Updates self.d_its with demand values for shelters and zero demand for non-shelter locations.
        """
        from demand_generation import in_sample, demand_deterministic_landfall, data
        track_err, intensity = s_t

        # Step 1: Get PoD-level demand based on hurricane state
        demand_vector = demand_deterministic_landfall(data, track_err, int(intensity), t_bar)
        scaled_pod_demand = self.population * np.array(demand_vector)

        # Step 2: Distribute demand to shelters using inverse-distance weights
        demand_df = pd.DataFrame({t_bar: scaled_pod_demand})
        shelter_demands = self.weights.multiply(demand_df[t_bar], axis=0).sum(axis=0).values

        # Step 3: Store in self.d_its for shelters
        for shelter_idx, i in enumerate(network_processor.V1):
            self.d_its[(i, t_bar, s_t)] = shelter_demands[shelter_idx]

        # Step 4: Assign 0 demand to all other (non-shelter) locations
        for i in range(network_processor.V):
            if i not in network_processor.V1:
                self.d_its[(i, t_bar, s_t)] = 0.0

    def rh_generate_future_demand(self, t_bar, s_t, t_L, network_processor, in_sample_paths):
        """
        Generate in-sample scenario-based shelter-level demand for t in [t_bar+1, t_L],
        using the current hurricane state s_t to create scenarios.
        
        Updates self.d_its with demand values for all in-sample scenarios.
        """
        from demand_generation import in_sample, demand_deterministic_landfall, data

        for s in range(self.num_sce):
            for t in range(t_bar + 1, t_L + 1):
                track_err_s, intensity_s = in_sample_paths[s, t]

                demand_vector = demand_deterministic_landfall(data, track_err_s, int(intensity_s), t)
                scaled_pod_demand = self.population * np.array(demand_vector)

                demand_df = pd.DataFrame({t: scaled_pod_demand})
                shelter_demands = self.weights.multiply(demand_df[t], axis=0).sum(axis=0).values

                for shelter_idx, i in enumerate(network_processor.V1):
                    self.d_its[(i, t, s)] = shelter_demands[shelter_idx]

                for i in range(network_processor.V):
                    if i not in network_processor.V1:
                        self.d_its[(i, t, s)] = 0.0

    def rh_generate_post_landfall_demand(self, t_L, in_sample_paths, network_processor):
        """
        Generate deterministic PoD-level demand for t > t_L (i.e., post-landfall),
        based on hurricane category at landfall time for each in-sample scenario.

        Updates self.d_its with demand values for PoD and non-PoD locations.
        """
        for s in range(self.num_sce):
            track_err_L, hurr_cat_L = in_sample_paths[s, t_L]
            cat = int(hurr_cat_L)

            if cat == 0 or pd.isna(cat):
                for i in network_processor.V3:
                    for t in network_processor.T_Aft:
                        self.d_its[(i, t, s)] = 0.0
            else:
                demand_col = f"I*p_c (cat {cat})"
                evac_col = "% 1- (Likely to evacuate)"
                total_demand = self.pod_info_df[demand_col] * self.pod_info_df[evac_col]
                total_demand = total_demand.to_numpy()

                mean = np.mean(network_processor.T_Aft)
                std = self.std_pod
                weights = np.array([
                    np.exp(-((t - mean) ** 2) / (2 * std ** 2)) for t in network_processor.T_Aft
                ])
                weights /= weights.sum()

                for idx, i in enumerate(network_processor.V3):
                    for w_idx, t in enumerate(network_processor.T_Aft):
                        if (i, t) not in self.d_its:
                            self.d_its[(i, t)] = total_demand[idx] * weights[w_idx]

                # Ensure zero demand for non-PoD locations
                for i in range(network_processor.V):
                    if i not in network_processor.V3:
                        for t in network_processor.T_Aft:
                            if (i, t) not in self.d_its:
                                self.d_its[(i, t)] = 0.0

    
    def generate_in_sample_demand(self, num_sce, network_processor, output_dir="csv_by_location"):
        """
        Generates in-sample demand based on hurricane track and category,
        scales by population, saves CSV files per location, and landfall states.

        Returns:
            in_sample_data: Array of in-sample hurricane states
        """
        from demand_generation import in_sample, demand_deterministic_landfall, data  # Local import

        if self.population is None:
            raise ValueError("Population data not loaded. Call load_oos_inputs() first.")

        initial_state = data.state_space[0][0]
        S = num_sce

        in_sample_data = in_sample(0, initial_state, S)

        # Save landfall states to Excel
        self.save_landfall_states(in_sample_data, network_processor)

        all_demands = []
        for s in range(S):
            sample_demands = []
            for t in network_processor.T_Bef:
                track_err, intensity = in_sample_data[s, t]
                demand = demand_deterministic_landfall(data, track_err, intensity, t)
                sample_demands.append(demand)
            all_demands.append(sample_demands)

        # Scale by population
        scaled_demands = np.array(all_demands) * self.population  # (S, T, I)

        # Save location-wise CSVs
        os.makedirs(output_dir, exist_ok=True)
        S, T, I = scaled_demands.shape

        for i in range(I):
            data_matrix = [np.round(scaled_demands[s, :, i], 3) for s in range(S)]
            df = pd.DataFrame(data_matrix)
            df.to_csv(os.path.join(output_dir, f"location_{i}.csv"), index=False, header=False)

        print(f"✔ Saved scaled demand CSVs to: {os.path.abspath(output_dir)}")

        return in_sample_data  # needed for next steps

    def allocate_demand(self, demand_files, output_file, network_processor, num_scenarios=None):

        # Allocates demand from PoDs to shelters based on inverse distance.

        if num_scenarios is None:
            num_scenarios = self.num_sce

        if self.weights is None:
            raise ValueError("Inverse distance weights not loaded. Call load_oos_inputs() first.")
        
        weights = self.weights

        # Load demand data (7 files, each with 1048 scenarios & 6 time periods)
        demand_scenarios = [pd.read_csv(file, header=None) for file in demand_files]

        # Allocate demand for the specified number of scenarios
        allocated_demand = {}
        for scenario_idx in range(num_scenarios):
            # Extract demand for this scenario across all 7 PODs (6 time periods each)
            demand_df = pd.DataFrame({t: [demand_scenarios[pod].iloc[scenario_idx, t] for pod in range(network_processor.num_pods)] for t in range(len(network_processor.T_Bef))})

            # Allocate demand based on distance weights
            allocation_matrix = {t: weights.multiply(demand_df[t], axis=0).sum(axis=0) for t in range(len(network_processor.T_Bef))}

            # Convert to DataFrame where rows are shelters, columns are time periods
            allocated_demand[f"Scenario_{scenario_idx}"] = pd.DataFrame(allocation_matrix, index=range(network_processor.num_shelters))

        # Save the results
        with pd.ExcelWriter(output_file) as writer:
            for scenario, df in allocated_demand.items():
                df.to_excel(writer, sheet_name=scenario)

        print(f"✅ Demand allocation completed. File saved: {output_file}")

    def save_landfall_states(self, in_sample_data, network_processor, output_file="Generated data/LandfallStates.xlsx"):
        """
        Extract hurricane track and category at time of landfall and save to Excel.

        """
        t_L = network_processor.t_L
        S = in_sample_data.shape[0]
        landfall_states = []

        for s in range(S):
            track_err, hurr_cat = in_sample_data[s, t_L]
            landfall_states.append([track_err, hurr_cat])

        df_landfall = pd.DataFrame(landfall_states, columns=["track_err", "hurr_cat"])
        df_landfall["hurr_cat"] = df_landfall["hurr_cat"].apply(
            lambda x: f"Category {int(x)}" if pd.notna(x) else "N/A")

        df_landfall.to_excel(output_file, index=False)
        print(f"✔ Saved PoD landfall states to: {output_file}")

    def calculate_landfall_demand(self, wind_speed_file, pod_info_file, output_file):
        # Load the data
        wind_speed_df = pd.read_excel(wind_speed_file, header=0)
        pod_info_df = pd.read_excel(pod_info_file, sheet_name='Sheet1')

        # Read hurricane categories from column 'hurr_cat'
        categories = wind_speed_df['hurr_cat'].fillna("N/A")

        # Define columns corresponding to each category
        category_columns = {
            "Category 1": "I*p_c (cat 1)",
            "Category 2": "I*p_c (cat 2)",
            "Category 3": "I*p_c (cat 3)",
            "Category 4": "I*p_c (cat 4)",
            "Category 5": "I*p_c (cat 5)"
        }

        # Get the column for '% 1- (Likely to evacuate)'
        evac_column = '% 1- (Likely to evacuate)'

        # Create an Excel writer to save the results
        with pd.ExcelWriter(output_file) as writer:
            for scenario_idx in range(len(categories)):
                # Get the category for this scenario
                category = categories.iloc[scenario_idx]

                # Check if the category is valid
                demand_column = category_columns.get(category, None)

                if demand_column is not None:
                    # Extract the necessary data for calculation
                    pod_demand_values = pod_info_df[demand_column]
                    evacuation_factors = pod_info_df[evac_column]

                    # Calculate demand: (I*p_c) * (% 1- Likely to evacuate)
                    demand_values = pod_demand_values * evacuation_factors

                else:
                    # If the category is N/A, set demand to 0
                    demand_values = pd.Series(0, index=pod_info_df.index)

                # Ensure numeric data type
                demand_values = demand_values.astype(float)

                # Prepare the output DataFrame
                scenario_data = pd.DataFrame({
                    'County': pod_info_df['County'],
                    'total demand': demand_values
                })

                # Write to Excel sheet for this scenario
                sheet_name = f'Scenario_{scenario_idx}'
                scenario_data.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"✅ Corrected output file generated: {output_file}")

    def distribute_demand_by_time(self, total_demand_file, network_processor, output_file):
        # Load the total demand file
        total_demand_data = pd.ExcelFile(total_demand_file)

        # Define time periods and normal distribution parameters
        time_periods = time_periods = network_processor.T_Aft
        mean = np.mean(network_processor.T_Aft)
        std = self.std_pod

        # Calculate normalized weights using a normal distribution
        distribution_values = [np.exp(-((t - mean) ** 2) / (2 * (std ** 2))) for t in time_periods]
        weights = np.array(distribution_values) / sum(distribution_values)  # Normalize weights to sum to 1

        # Create an Excel writer to save the results
        with pd.ExcelWriter(output_file) as writer:
            for sheet_name in total_demand_data.sheet_names:
                # Read the data for the current scenario
                scenario_data = pd.read_excel(total_demand_data, sheet_name=sheet_name)

                # Apply the distribution weights to the total demand for each county
                distributed_demands = {
                    f'Time_{t}': scenario_data['total demand'] * weight for t, weight in zip(time_periods, weights)
                }

                # Prepare the output DataFrame
                output_data = scenario_data[['County']].copy()
                for period, demand in distributed_demands.items():
                    output_data[period] = demand

                # Write the output data to the Excel sheet
                output_data.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"✅ Distributed demand file generated: {output_file}")
    
    def combine_demand_data(self, shelter_file, pod_file, network_processor, output_file, num_scenarios=None):
        """
        Combine shelter and pod demand into one Excel file with scenario sheets.
        """
        if num_scenarios is None:
            num_scenarios = self.num_sce

        # Read data from both files (assuming data is in each scenario sheet)
        shelter_data = pd.read_excel(shelter_file, sheet_name=None)
        pod_data = pd.read_excel(pod_file, sheet_name=None)

        # Create an output Excel writer using openpyxl
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for scenario_num in range(num_scenarios):
                scenario_sheet = f'Scenario_{scenario_num}'

                # Get data for the current scenario
                shelter_scenario = shelter_data.get(scenario_sheet, pd.DataFrame())
                pod_scenario = pod_data.get(scenario_sheet, pd.DataFrame())

                # Initialize combined data structure
                combined_data = pd.DataFrame(0.0, index=range(network_processor.num_locations), columns=range(network_processor.T))  # Initialize with float values

                # Assign shelter demand
                if not shelter_scenario.empty:
                    shelter_values = shelter_scenario.iloc[:, 1 : 1 + len(network_processor.T_Bef)].values
                    for row_idx, loc_id in enumerate(network_processor.V1):
                        combined_data.iloc[loc_id, 0 : len(network_processor.T_Bef)] = shelter_values[row_idx]

                # Assign PoD demand
                if not pod_scenario.empty:
                    pod_values = pod_scenario.iloc[:, 1 : 1 + len(network_processor.T_Aft)].values
                    for row_idx, loc_id in enumerate(network_processor.V3):
                        combined_data.iloc[loc_id, len(network_processor.T_Bef) : ] = pod_values[row_idx]

                # Add index column
                combined_data.insert(0, 'Index', range(1, network_processor.num_locations + 1))

                # Write combined data to a new sheet
                combined_data.to_excel(writer, sheet_name=scenario_sheet, index=False)

        print(f"✅Combined data saved to {output_file}")

    def compute_expected_demand(self):

        expected_demand = {}  # Dictionary to store averaged demand

        # Iterate over all locations and time periods
        for (i, t, s) in self.d_its:
            if (i, t) not in expected_demand:
                expected_demand[(i, t)] = 0  # Initialize sum
        
            expected_demand[(i, t)] += self.d_its[(i, t, s)] / self.num_sce

        self.d_its_expected = {(i, t, 0): expected_demand[(i, t)] for (i, t) in expected_demand}

    


