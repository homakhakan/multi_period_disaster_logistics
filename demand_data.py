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

    def load_input_files(self, population_file, distance_file, network_processor):
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

    def generate_current_demand(self, t_bar, s_t, network_processor, data, solution_method):
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

        scenario_key = 0 if solution_method == 1 else s_t

        for shelter_idx, i in enumerate(network_processor.V1):
            self.d_its[(i, t_bar, scenario_key)] = shelter_demands[shelter_idx]

        for i in range(network_processor.V):
            if i not in network_processor.V1:
                self.d_its[(i, t_bar, scenario_key)] = 0.0

    def generate_future_demand(self, t_bar, s_t, t_L, network_processor, in_sample_paths, solution_method):
        """
        Generate in-sample scenario-based shelter-level demand for t in [t_bar+1, t_L],
        using the current hurricane state s_t to create scenarios.
        
        Updates self.d_its with demand values for all in-sample scenarios.
        """
        from demand_generation import in_sample, demand_deterministic_landfall, data

        if solution_method == 1:
            # Static 2SSP: use only the realized path (s=0)
            for t in range(t_bar + 1, t_L + 1):
                track_err_s, intensity_s = in_sample_paths[0, t]

                demand_vector = demand_deterministic_landfall(data, track_err_s, int(intensity_s), t)
                scaled_pod_demand = self.population * np.array(demand_vector)

                demand_df = pd.DataFrame({t: scaled_pod_demand})
                shelter_demands = self.weights.multiply(demand_df[t], axis=0).sum(axis=0).values

                for shelter_idx, i in enumerate(network_processor.V1):
                    self.d_its[(i, t, 0)] = shelter_demands[shelter_idx]

                for i in range(network_processor.V):
                    if i not in network_processor.V1:
                        self.d_its[(i, t, 0)] = 0.0

        else:
            # Rolling Horizon: multiple scenarios
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

    def generate_post_landfall_demand(self, t_L, in_sample_paths, network_processor, solution_method):
        """
        Generate deterministic PoD-level demand for t > t_L (i.e., post-landfall),
        based on hurricane category at landfall time for each in-sample scenario.

        Updates self.d_its with demand values for PoD and non-PoD locations.
        """

        if solution_method == 1:  # Static 2SSP
            # Just use the first (realized) path
            track_err_L, hurr_cat_L = in_sample_paths[0, t_L]
            cat = int(hurr_cat_L)

            if cat == 0 or pd.isna(cat):
                for i in network_processor.V3:
                    for t in network_processor.T_Aft:
                        self.d_its[(i, t, 0)] = 0.0
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
                        self.d_its[(i, t, 0)] = total_demand[idx] * weights[w_idx]

            # Ensure zero demand for non-PoDs
            for i in range(network_processor.V):
                if i not in network_processor.V3:
                    for t in network_processor.T_Aft:
                        self.d_its[(i, t, 0)] = 0.0

        else:  # Rolling Horizon
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
    
    def save_landfall_states(self, in_sample_data, network_processor, output_file="Generated data/LandfallStates.xlsx", debug_mode=False):
        """
        Extract hurricane track and category at time of landfall.
        If debug_mode=True, save to Excel. Otherwise, return the DataFrame.
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

        if debug_mode:
            df_landfall.to_excel(output_file, index=False)
            print(f"✅ Saved PoD landfall states to: {output_file}")
        else:
            return df_landfall

    def generate_in_sample_demand(self, num_sce, network_processor, output_dir="csv_by_location", debug_mode=False):
        """
        Generates in-sample demand based on hurricane track and category,
        scales by population. In debug_mode, saves files for inspection.

        Returns:
            Tuple: (in_sample_data, scaled_demands) if debug_mode=False,
                in_sample_data only if debug_mode=True
        """
        from demand_generation import in_sample, demand_deterministic_landfall, data

        if self.population is None:
            raise ValueError("Population data not loaded. Call load_input_files() first.")

        initial_state = data.state_space[0][0]
        S = num_sce

        in_sample_data = in_sample(0, initial_state, S)

        # Save or return landfall states
        self.save_landfall_states(in_sample_data, network_processor, debug_mode=debug_mode)

        # Generate demand matrix
        all_demands = []
        for s in range(S):
            sample_demands = []
            for t in network_processor.T_Bef:
                track_err, intensity = in_sample_data[s, t]
                demand = demand_deterministic_landfall(data, track_err, intensity, t)
                sample_demands.append(demand)
            all_demands.append(sample_demands)

        # Scale by population
        scaled_demands = np.array(all_demands) * self.population  # Shape: (S, T, I)

        if debug_mode:
            os.makedirs(output_dir, exist_ok=True)
            S, T, I = scaled_demands.shape

            for i in range(I):
                data_matrix = [np.round(scaled_demands[s, :, i], 3) for s in range(S)]
                df = pd.DataFrame(data_matrix)
                df.to_csv(os.path.join(output_dir, f"location_{i}.csv"), index=False, header=False)

            print(f"✅ Saved scaled demand CSVs to: {os.path.abspath(output_dir)}")
            return in_sample_data

        else:
            return in_sample_data, scaled_demands

    def allocate_demand(self, demand_data=None, demand_files=None, output_file=None, network_processor=None, num_scenarios=None, debug_mode=False):
        """
        Allocates demand from PoDs to shelters based on inverse distance weights.
        When debug_mode is True, saves Excel output; otherwise returns the allocation in memory.

        Returns:
            If debug_mode is False: dict of {scenario_name: DataFrame}
        """
        if num_scenarios is None:
            num_scenarios = self.num_sce

        if self.weights is None:
            raise ValueError("Inverse distance weights not loaded. Call load_input_files() first.")

        weights = self.weights

        allocated_demand = {}

        if debug_mode:
            if demand_files is None:
                raise ValueError("In debug mode, demand_files must be provided.")
            demand_scenarios = [pd.read_csv(file, header=None) for file in demand_files]

            for scenario_idx in range(num_scenarios):
                demand_df = pd.DataFrame({
                    t: [demand_scenarios[pod].iloc[scenario_idx, t] for pod in range(network_processor.num_pods)]
                    for t in range(len(network_processor.T_Bef))
                })
                allocation_matrix = {
                    t: weights.multiply(demand_df[t], axis=0).sum(axis=0) for t in range(len(network_processor.T_Bef))
                }
                df = pd.DataFrame(allocation_matrix, index=range(network_processor.num_shelters))
                expected_shape = (network_processor.num_shelters, len(network_processor.T_Bef))
                if df.shape != expected_shape:
                    print(f"⚠️ Scenario {scenario_idx} (debug): Shape mismatch {df.shape}, expected {expected_shape}")
                    print(df.head())

                allocated_demand[f"Scenario_{scenario_idx}"] = df

            if output_file is None:
                raise ValueError("output_file must be specified when debug_mode=True")
            with pd.ExcelWriter(output_file) as writer:
                for scenario, df in allocated_demand.items():
                    df.to_excel(writer, sheet_name=scenario)
            print(f"✅ Demand allocation completed. File saved: {output_file}")

        else:
            if demand_data is None:
                raise ValueError("In non-debug mode, demand_data must be provided.")

            S, T, I = demand_data.shape
            for s in range(S):
                scenario_demand = demand_data[s]  # shape (T, I)

                demand_df = pd.DataFrame({
                    t: [scenario_demand[t, pod] for pod in range(network_processor.num_pods)]
                    for t in range(T)
                })

                allocation_matrix = {
                    t: weights.multiply(demand_df[t], axis=0).sum(axis=0) for t in range(T)
                }
                df = pd.DataFrame(allocation_matrix, index=range(network_processor.num_shelters))
                expected_shape = (network_processor.num_shelters, T)
                if df.shape != expected_shape:
                    print(f"⚠️ Scenario {s}: Shape mismatch {df.shape}, expected {expected_shape}")
                    print(df.head())

                allocated_demand[f"Scenario_{s}"] = df

            print("✅ Demand allocation completed in memory.")
            return allocated_demand

    def calculate_landfall_demand(self, wind_speed_file=None, landfall_df=None, pod_info_file=None, output_file=None, debug_mode=False):
        """
        Calculate total PoD demand after landfall using hurricane categories.
        - If debug_mode is True, saves to Excel (as before).
        - If debug_mode is False, returns a dict of DataFrames in memory.
        """
        if debug_mode:
            wind_speed_df = pd.read_excel(wind_speed_file, header=0)
        else:
            if landfall_df is None:
                raise ValueError("landfall_df must be provided when debug_mode=False")
            wind_speed_df = landfall_df

        pod_info_df = pd.read_excel(pod_info_file, sheet_name='Sheet1')
        categories = wind_speed_df['hurr_cat'].fillna("N/A")

        category_columns = {
            "Category 1": "I*p_c (cat 1)",
            "Category 2": "I*p_c (cat 2)",
            "Category 3": "I*p_c (cat 3)",
            "Category 4": "I*p_c (cat 4)",
            "Category 5": "I*p_c (cat 5)"
        }
        evac_column = '% 1- (Likely to evacuate)'

        demand_results = {}

        for scenario_idx in range(len(categories)):
            category = categories.iloc[scenario_idx]
            demand_column = category_columns.get(category, None)

            if demand_column is not None:
                pod_demand_values = pod_info_df[demand_column]
                evacuation_factors = pod_info_df[evac_column]
                demand_values = pod_demand_values * evacuation_factors
            else:
                demand_values = pd.Series(0, index=pod_info_df.index)

            demand_values = demand_values.astype(float)

            scenario_data = pd.DataFrame({
                'County': pod_info_df['County'],
                'total demand': demand_values
            })

            sheet_name = f'Scenario_{scenario_idx}'
            demand_results[sheet_name] = scenario_data

        if debug_mode:
            if output_file is None:
                raise ValueError("output_file must be specified when debug_mode=True")

            with pd.ExcelWriter(output_file) as writer:
                for sheet_name, df in demand_results.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"✅ Corrected output file generated: {output_file}")
        else:
            print("✅ Landfall demand generated in memory.")
            return demand_results

    def distribute_demand_by_time(self, total_demand_file=None, total_demand_dict=None, network_processor=None, output_file=None, debug_mode=False):
        """
        Distribute total post-landfall demand across time periods using a normal distribution.

        If debug_mode is True, saves to Excel (as before).
        If debug_mode is False, returns a dictionary of DataFrames.

        Returns:
            dict (only if debug_mode is False): {sheet_name: DataFrame}
        """
        if debug_mode:
            if total_demand_file is None:
                raise ValueError("total_demand_file must be provided when debug_mode=True")
            total_demand_data = pd.ExcelFile(total_demand_file)
            sheet_names = total_demand_data.sheet_names
        else:
            if total_demand_dict is None:
                raise ValueError("total_demand_dict must be provided when debug_mode=False")
            total_demand_data = total_demand_dict
            sheet_names = total_demand_data.keys()

        time_periods = network_processor.T_Aft
        mean = np.mean(time_periods)
        std = self.std_pod

        distribution_values = [np.exp(-((t - mean) ** 2) / (2 * (std ** 2))) for t in time_periods]
        weights = np.array(distribution_values) / sum(distribution_values)

        all_distributed = {}

        for sheet_name in sheet_names:
            if debug_mode:
                scenario_data = pd.read_excel(total_demand_data, sheet_name=sheet_name)
            else:
                scenario_data = total_demand_data[sheet_name]

            distributed_demands = {
                f'Time_{t}': scenario_data['total demand'] * weight
                for t, weight in zip(time_periods, weights)
            }

            output_data = scenario_data[['County']].copy()
            for period, demand in distributed_demands.items():
                output_data[period] = demand

            all_distributed[sheet_name] = output_data

        if debug_mode:
            if output_file is None:
                raise ValueError("output_file must be specified when debug_mode=True")

            with pd.ExcelWriter(output_file) as writer:
                for sheet_name, df in all_distributed.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"✅ Distributed demand file generated: {output_file}")
        else:
            print("✅ Distributed demand computed in memory.")
            return all_distributed

    def combine_demand_data(self, shelter_file=None, pod_file=None, shelter_data_dict=None, pod_data_dict=None, network_processor=None, output_file=None, num_scenarios=None, debug_mode=False):
        """
        Combine shelter and pod demand into one Excel file or return combined data per scenario.

        Args:
            debug_mode (bool): If True, writes to Excel. If False, returns dict of combined DataFrames.

        Returns:
            dict (if debug_mode is False): {scenario_name: combined DataFrame}
        """
        if num_scenarios is None:
            num_scenarios = self.num_sce

        if debug_mode:
            # Read Excel files
            shelter_data = pd.read_excel(shelter_file, sheet_name=None)
            pod_data = pd.read_excel(pod_file, sheet_name=None)
        else:
            # Use provided dictionaries
            if shelter_data_dict is None or pod_data_dict is None:
                raise ValueError("Both shelter_data_dict and pod_data_dict must be provided when debug_mode=False")
            shelter_data = shelter_data_dict
            pod_data = pod_data_dict

        combined_results = {}

        for scenario_num in range(num_scenarios):
            scenario_sheet = f'Scenario_{scenario_num}'

            shelter_scenario = shelter_data.get(scenario_sheet, pd.DataFrame())
            pod_scenario = pod_data.get(scenario_sheet, pd.DataFrame())

            combined_data = pd.DataFrame(0.0, index=range(network_processor.num_locations), columns=range(network_processor.T))

            if not shelter_scenario.empty:
                shelter_values = shelter_scenario.iloc[:, :len(network_processor.T_Bef)].values
                for row_idx, loc_id in enumerate(network_processor.V1):
                    if shelter_values.shape[1] != len(network_processor.T_Bef):
                        raise ValueError(
                            f"Scenario {scenario_sheet}: Expected {len(network_processor.T_Bef)} shelter columns, got {shelter_values.shape[1]}"
                        )
                    combined_data.iloc[loc_id, 0 : len(network_processor.T_Bef)] = shelter_values[row_idx]

            if not pod_scenario.empty:
                pod_values = pod_scenario.iloc[:, 1 : 1 + len(network_processor.T_Aft)].values
                for row_idx, loc_id in enumerate(network_processor.V3):
                    if pod_values.shape[1] != len(network_processor.T_Aft):
                        raise ValueError(
                            f"Scenario {scenario_sheet}: Expected {len(network_processor.T_Aft)} PoD columns, got {pod_values.shape[1]}"
                        )
                    combined_data.iloc[loc_id, len(network_processor.T_Bef) : ] = pod_values[row_idx]

            combined_data.insert(0, 'Index', range(1, network_processor.num_locations + 1))
            combined_results[scenario_sheet] = combined_data

        if debug_mode:
            if output_file is None:
                raise ValueError("output_file must be provided in debug mode.")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for sheet_name, df in combined_results.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"✅ Combined data saved to {output_file}")
        else:
            print("✅ Combined demand data prepared in memory.")
            return combined_results

    def populate_demand(self, combined_demand_dict):
        """
        Populate self.d_its using combined demand data dictionary and update scenario list and probabilities.
        """
        for scenario_name, df in combined_demand_dict.items():
            scenario_idx = int(scenario_name.split("_")[-1])  # Extract s from "Scenario_0"
            for i in range(df.shape[0]):
                for t in range(1, df.shape[1]):  # Skip 'Index' column
                    loc_id = i
                    time_step = t - 1
                    demand_value = df.iloc[i, t]
                    self.d_its[(loc_id, time_step, scenario_idx)] = demand_value

        #for (i, t, s), value in self.d_its.items():
         #   if value != 0:
          #      print(f"(i={i}, t={t}, s={s}) → demand = {value}")

        self.sce = list(range(self.num_sce))
        self.P = [1 / self.num_sce] * self.num_sce

    def compute_expected_demand(self):

        expected_demand = {}  # Dictionary to store averaged demand

        # Iterate over all locations and time periods
        for (i, t, s) in self.d_its:
            if (i, t) not in expected_demand:
                expected_demand[(i, t)] = 0  # Initialize sum
        
            expected_demand[(i, t)] += self.d_its[(i, t, s)] / self.num_sce

        self.d_its_expected = {(i, t, 0): expected_demand[(i, t)] for (i, t) in expected_demand}

    def msp_demand_mapping(self, t, k_t):
        """Return a dict {i: demand_i(t)} based on realized state k_t."""

        from demand_generation import demand_deterministic_landfall, data

        track_err, intensity = k_t

        # PoD-level demand (existing helper)
        demand_vector = demand_deterministic_landfall(data, track_err, int(intensity), t)
        scaled_pod = self.population * np.array(demand_vector)

        # Map PoD demand to shelters with inverse-distance weights
        demand_df = pd.DataFrame({t: scaled_pod})
        shelter_demands = (self.weights.multiply(demand_df[t], axis=0).sum(axis=0).values)

        # Build node-level dict
        demand_i = {}
        for idx, i in enumerate(self.V1):   # shelters
            demand_i[i] = shelter_demands[idx]
        for i in range(self.V):             # non-shelters
            if i not in self.V1:
                demand_i[i] = 0.0
        return demand_i


