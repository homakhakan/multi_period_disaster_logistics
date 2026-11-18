import pandas as pd
import numpy as np
import math
import json

class NetworkData:
    def __init__(self, instance_version, params):

        # Read MC inputs
        with open("mc_data/inputs.json", "r") as f:
            raw = json.load(f)
        mc_args = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}
        H = int(mc_args["H"])
        segments = int(mc_args["segments"])
   
        self.instance_version = instance_version
        self.params = params
        self.tperiod = int(12 / segments)
        self.thetaLB = float(params["THETA_LB"])
        self.eps = float(params["EPS"])
        self.T1 = int(H / self.tperiod)
        self.T2 = math.ceil((0.6 * H) / self.tperiod) # Relative to T1
        self.total_hours = (self.T1 + self.T2) * self.tperiod
        self.fuel_cost_per_mile = float(params["FUEL_COST_PER_MILE"])
        self.base_cap = int(params["BASE_CAP"])
        # print("T1:", self.T1)
        # print("T2:", self.T2)
        # print("t_period:", self.tperiod)
        # exit(0)

        # Load file paths
        self.get_file_paths()

        # Load data files
        self.df_V = pd.read_excel(self.file_paths["location_file"])
        self.df_R = pd.read_excel(self.file_paths["resource_file"])

        # Load matrices
        self.durationMatrix = np.loadtxt(self.file_paths["duration_matrix_file"])
        self.distanceMatrix = np.loadtxt(self.file_paths["distance_matrix_file"])

    @property
    def num_warehouses(self):
        return len(self.V0)

    @property
    def num_shelters(self):
        return len(self.V1)

    @property
    def num_RSAs(self):
        return len(self.V2)

    @property
    def num_pods(self):
        return len(self.V3)

    @property
    def num_locations(self):
        return len(self.df_V)


    
    def get_file_paths(self): # Times in duration matrix is in "seconds"
    
        self.file_paths = {
            "location_file": f"Data/Set V_locations_{self.instance_version}.xlsx",
            "resource_file": f"Data/Set R_resources.xlsx",
            "duration_matrix_file": f"Data/Duration_all_nodes_{self.instance_version}.txt",
            "distance_matrix_file": f"Data/Distance_all_nodes_{self.instance_version}.txt",
        }

    def define_sets(self):

        self.V = len(self.df_V)  # Number of locations
        self.R = len(self.df_R)  # Number of resources
        self.T = int(self.total_hours / self.tperiod)  # Compute time periods

        # Subsets of Locations
        self.V0, self.V1, self.V2, self.V3 = [], [], [], []
        for i in range(self.V):
            location_type = self.df_V['Type'][i]
            index = int(self.df_V['Index'][i])
            if location_type in ['Warehouse', 'Supplier']:
                self.V0.append(index)
            elif location_type == 'Shelter':
                self.V1.append(index)
            elif location_type == 'RSA':
                self.V2.append(index)
            elif location_type == 'PoD':
                self.V3.append(index)    

        # Time period subsets
        self.T_Bef = list(range(self.T1))
        self.T_Aft = list(range(self.T1, self.T))

        # Time of landfall
        self.t_L = self.T1 - 1 # Index of landfall

        self.A = self.create_arc_set()

    def create_arc_set(self):
        self.A = set()

        # nodes with no demand
        no_demand = set()

        # classify V0 dynamically from the Type column
        W = {i for i in self.V0 if str(self.df_V.at[i, 'Type']).lower() == 'warehouse'}
        S = {i for i in self.V0 if str(self.df_V.at[i, 'Type']).lower() == 'supplier'}

        # build arcs
        for i in range(self.V):
            if i in no_demand:
                continue
            for j in range(self.V):
                if j == i or j in no_demand:
                    continue
                # no Shelter <-> PoD in either direction
                if (i in self.V1 and j in self.V3) or (i in self.V3 and j in self.V1):
                    continue
                # If origin is Shelter, only allow Shelter→Shelter
                if i in self.V1 and j not in self.V1:
                    continue

                # If origin is PoD, only allow PoD→PoD
                if i in self.V3 and j not in self.V3:
                    continue
                # RSA -> only Shelter or PoD, or another RSA (no self-loop)
                if i in self.V2:
                    if not (j in self.V1 or j in self.V3 or (j in self.V2 and j != i)):
                        continue
                # V0 internal: allow only Supplier -> Warehouse
                if i in self.V0 and j in self.V0 and not (i in S and j in W):
                    continue
                self.A.add((i, j))
        
        # for i, j in sorted(self.A, key=lambda x: (x[0], x[1])):
        #     print((i, j))

        return self.A

    def define_parameters(self, travel_time):
        
        self.fuel = self.fuel_cost_per_mile * self.params["MILES_TO_METERS"] # Fuel cost per meter (converted from miles)    

        # Initialize inventory list
        # Filter inventory columns based on selected commodity (water, food, or both)
        selected_columns = [c for c in self.params["INVENTORY_COLUMNS"][self.instance_version] if self.commodity == "both" or self.commodity.lower() in c.lower()]
        # print(selected_columns)

        
        Inv = []
        for i in range(len(self.df_V)):
            Inv.extend([self.df_V[col][i] for col in selected_columns])

        self.R = len(selected_columns)  # Number of resources
        # print("R=", self.R)

        self.multiplier_init_inv = float(self.params["MULTIPLIER_INIT_INV"])
        self.I_ir0 = [[value * self.multiplier_init_inv for value in Inv[i:i+self.R]] for i in range(0, len(Inv), self.R)]
        # print(self.I_ir0)

        # Penalty costs per location
        self.multiplier_penalty_cost = float(self.params["MULTIPLIER_PENALTY_COST"])
        self.p_i = (self.df_V["P_i"] * self.multiplier_penalty_cost * (self.tperiod / 12)).tolist() # Scaling penalty cost relative to different time intervals

        if travel_time == 1:
            # Travel time in time periods (travel times are only 1 time period)
            self.t_ij = np.ones((self.V, self.V), dtype=int)  # fill with 1s
            np.fill_diagonal(self.t_ij, 0)  # set diagonal to 0
            # Loop through durationMatrix to find off-diagonal zeros
            for i in range(self.V):
                for j in range(self.V):
                    if i != j and self.durationMatrix[i][j] == 0:
                        self.t_ij[i][j] = 0

            # print(self.t_ij)
            # exit(0)

        elif travel_time == 2:
            # Multi-period travel time computed from real duration matrix
            # durationMatrix is in seconds
            # tperiod is in hours (3, 6)

            dur_seconds = np.array(self.durationMatrix, dtype=float)
            dur_hours = (dur_seconds / 3600) * 1.75 # #Hardcode! # Convert seconds to hours # scaling factor to include loading/unloading/etc. 
            self.t_ij = np.ceil(dur_hours / self.tperiod).astype(int) # Compute travel time in number of time periods
            np.fill_diagonal(self.t_ij, 0) # Keep diagonal = 0

            # If any off-diagonal duration is 0, keep it 0
            for i in range(self.V):
                for j in range(self.V):
                    if i != j and dur_seconds[i][j] == 0:
                        self.t_ij[i][j] = 0
            # print(self.t_ij)
            # exit(0)

            self.tt_max = int(np.max(self.t_ij)) # Extract the maximum travel time (without delay) to later use in sddp for saving some computational time

        #Starting number of trucks at location i (time period 0)
        self.C_i0 = self.df_V["TruckStart"].tolist()
        
        # Truck delay per location
        self.h_i = self.df_V['TruckDelay'].tolist()

        # Travel cost between locations
        self.m_ij = [[(self.distanceMatrix[i][j] * self.fuel) for j in range(len(self.df_V))] for i in range(len(self.df_V))]
        # print(self.m_ij)
        # exit(0)

        # Procurement cost per resource
        self.cost_ir = []
        idx = self.df_V[self.df_V['Type'].isin(['Warehouse', 'Supplier'])].index
        for i in range(len(self.df_V)):
            if i in idx:
                water_pc = float(self.df_V.at[i, 'Water PC'])
                food_pc = float(self.df_V.at[i, 'Food PC'])
                self.cost_ir.append([water_pc, food_pc] if self.commodity == "both" else [water_pc if self.commodity == "water" else food_pc])
        # print(self.cost_ir)

        # Fraction of a truck 1 pallet of resource r requires
        self.w_r = [1 / pallet for pallet, r in zip(self.df_R['Pallet Units on Truck'], self.df_R['R']) 
                    if self.commodity == "both" or self.commodity.lower() == r.lower()]
        # print("w_r:", self.w_r)
        
  
        # Factor of pallet required per commodity r
        self.g_shelter = [g for g, r in zip(self.df_R['pallet per S-Kit'], self.df_R['R']) 
                          if self.commodity == "both" or self.commodity.lower() == r.lower()]
        self.g_pod = [g for g, r in zip(self.df_R['pallet per P-Kit'], self.df_R['R']) 
                      if self.commodity == "both" or self.commodity.lower() == r.lower()]
        # print("g_shelter:", self.g_shelter)
        # print("g_pod:", self.g_pod)

        # Initialize K_ijt
        self.multiplier_arc = float(self.params["MULTIPLIER_ARC"])  # Scaling factor
        self.K_ijt = {}
        for (i, j) in self.A:
            self.K_ijt.setdefault(i, {})[j] = {}
            for t in range(self.T):
                self.K_ijt[i][j][t] = self.base_cap * self.multiplier_arc


    def expand_network(self):
        """Expand the original network by replacing each arc with travel time 
        greater than one period with a chain of auxiliary waypoint nodes and 
        unit-travel-time arcs."""

        exp_arcs = [(i, j, self.t_ij[i][j]) for (i, j) in self.A if self.t_ij[i][j] > 1]
        # for i, j, tt in exp_arcs:
        #     print(f"({i}, {j}) -> travel time = {tt}")

        # Initialize expanded network structures from the original network
        self.A_exp = set(self.A)
        self.V_exp = self.V
        self.df_V_exp = self.df_V.copy()
        
        # Remove each long arc from the expanded arc set before replacing it with unit-time arcs
        self.aux_map = {}   # Mapping auxiliary node IDs to readable labels for debugging
        for i, j, tt in exp_arcs:
            self.A_exp.remove((i, j))
            # print(f"Removed long arc ({i}, {j}) with travel time {tt}")

            # Create auxiliary waypoint nodes for this long arc
            aux_nodes = []
            for k in range(1, tt):
                aux_idx = self.V_exp
                self.V_exp += 1
                aux_nodes.append(aux_idx)
                self.aux_map[aux_idx] = f"aux_{i}_{j}_{k}"   # Mapping auxiliary node IDs to readable labels for debugging

                # Add auxiliary node to expanded location dataframe
                aux_row = pd.Series(index=self.df_V.columns, dtype=object)
                aux_row["Index"] = aux_idx
                aux_row["Type"] = "Aux"
                
                # Set NaN fields
                aux_row["Facility"] = np.nan
                aux_row["Address"] = np.nan
                aux_row["Lat"] = np.nan
                aux_row["Long"] = np.nan
                aux_row["Water PC"] = np.nan
                aux_row["Food PC"] = np.nan

                # Set zero fields
                aux_row["Water StartInv"] = 0
                aux_row["Food StartInv"] = 0
                aux_row["Cot StartInv"] = 0
                aux_row["Blanket StartInv"] = 0
                aux_row["TruckStart"] = 0
                aux_row["TruckDelay"] = 0
                aux_row["P_i"] = 0

                self.df_V_exp.loc[aux_idx] = aux_row
            
            # print(f"Auxiliary nodes for ({i}, {j}): {aux_nodes}")
            # print(f"Auxiliary labels: {[self.aux_map[a] for a in aux_nodes]}")

            # Create replacement chain: i -> aux_1 -> ... -> aux_last -> j
            chain_nodes = [i] + aux_nodes + [j]
            for a, b in zip(chain_nodes[:-1], chain_nodes[1:]):
                self.A_exp.add((a, b))
            # print(f"Replacement chain for ({i}, {j}): {chain_nodes}")

        # # Build expanded unit-travel-time matrix
        # self.t_ij_exp = np.zeros((self.V_exp, self.V_exp), dtype=int)
        # for i, j in self.A_exp:
        #     self.t_ij_exp[i][j] = 1

        # Build expanded travel-time matrix (temporary hack)
        self.t_ij_exp = np.zeros((self.V_exp, self.V_exp), dtype=int)

        # First assign t=1 to arcs that were not expanded
        for i, j in self.A:
            if (i, j) in self.A_exp:
                self.t_ij_exp[i][j] = 1

        # For expanded chains:
        # all internal arcs get t=0,
        # only the last arc gets t=1
        for i, j, tt in exp_arcs:

            aux_nodes = [aux_idx for aux_idx, label in self.aux_map.items() if label.startswith(f"aux_{i}_{j}_")]
            aux_nodes = sorted(aux_nodes, key=lambda a: int(self.aux_map[a].split("_")[-1]))
            chain_nodes = [i] + aux_nodes + [j]

            # Set all arcs to t=0
            for a, b in zip(chain_nodes[:-1], chain_nodes[1:]):
                self.t_ij_exp[a][b] = 0

            # Last arc gets t=1
            a_last, b_last = chain_nodes[-2], chain_nodes[-1]
            self.t_ij_exp[a_last][b_last] = 1

        # Build expanded transportation cost matrix
        old_m_ij = np.array(self.m_ij, dtype=float)
        self.m_ij_exp = np.zeros((self.V_exp, self.V_exp), dtype=float)
        # Keep original costs for arcs that were not expanded
        for i, j in self.A:
            if (i, j) in self.A_exp:
                self.m_ij_exp[i][j] = old_m_ij[i][j]
        
        # Assign original long-arc cost to the first arc of each replacement chain
        for i, j, tt in exp_arcs:
            aux_nodes = [aux_idx for aux_idx, label in self.aux_map.items() if label.startswith(f"aux_{i}_{j}_")]  # Find auxiliary nodes belonging to this original arc
            aux_nodes = sorted(aux_nodes, key=lambda a: int(self.aux_map[a].split("_")[-1]))                       # Sort auxiliary nodes in travel order
            chain_nodes = [i] + aux_nodes + [j]                                                                    # Reconstruct full replacement chain
            first_arc = (chain_nodes[0], chain_nodes[1])                                                           # Identify the first arc in the chain
            self.m_ij_exp[first_arc[0]][first_arc[1]] = old_m_ij[i][j]                                             # Put original long-arc cost on first replacement arc

        # Initialize expanded arc capacity dictionary
        self.K_ijt_exp = {}
        for (i, j) in self.A_exp:
            self.K_ijt_exp.setdefault(i, {})[j] = {}
            for t in range(self.T):
                self.K_ijt_exp[i][j][t] = self.base_cap

        self.V_aux = list(self.aux_map.keys())

        # Expand initial inventory for auxiliary nodes
        self.I_ir0_exp = [row.copy() for row in self.I_ir0]    # Copy original inventories
        for aux_idx in self.V_aux:
            self.I_ir0_exp.append([0 for _ in range(self.R)])  # Adds one zero-inventory row for each auxiliary node

        # Expand penalty cost for auxiliary nodes
        self.p_i_exp = list(self.p_i)
        for aux_idx in self.V_aux:
            self.p_i_exp.append(0)

        # Replace original network objects with expanded network objects
        self.V = self.V_exp
        self.A = self.A_exp
        self.df_V = self.df_V_exp
        self.t_ij = self.t_ij_exp
        self.m_ij = self.m_ij_exp
        self.K_ijt = self.K_ijt_exp
        self.I_ir0 = self.I_ir0_exp
        self.p_i = self.p_i_exp

