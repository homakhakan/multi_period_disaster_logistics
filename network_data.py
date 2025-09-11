import pandas as pd
import numpy as np
import math

class NetworkData:
    def __init__(self, instance_version, params):
   
        self.instance_version = instance_version
        self.params = params
        self.tperiod = int(params["TPERIOD"])
        self.thetaLB = float(params["THETA_LB"])
        self.eps = float(params["EPS"])
        self.T1 = int(params["T1"])
        self.T2 = int(params["T2"])
        self.total_hours = int(params["TOTAL_HOURS"])
        self.fuel_cost_per_mile = float(params["FUEL_COST_PER_MILE"])

        # Load file paths
        self.get_file_paths()

        # Load data files
        self.df_V = pd.read_excel(self.file_paths["location_file"])
        self.df_R = pd.read_excel(self.file_paths["resource_file"])
        self.df_K = pd.read_excel(self.file_paths["arc_capacity_file"])

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
            "resource_file": f"Data/Set R_resources_{self.instance_version}.xlsx",
            "duration_matrix_file": f"Data/Duration_all_nodes_{self.instance_version}.txt",
            "distance_matrix_file": f"Data/Distance_all_nodes_{self.instance_version}.txt",
            "arc_capacity_file": f"Data/Arc_capacity_{self.instance_version}.xlsx",
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

        return self.A

    def define_parameters(self, travel_time):
        
        self.fuel = self.fuel_cost_per_mile * self.params["MILES_TO_METERS"] # Fuel cost per meter (converted from miles)    

        # Initialize inventory list
        selected_columns = self.params["INVENTORY_COLUMNS"][self.instance_version]
        # print(selected_columns)

        
        Inv = []
        for i in range(len(self.df_V)):
            Inv.extend([self.df_V[col][i] for col in selected_columns])

        self.R = len(selected_columns)  # Number of resources
        # print("R=", self.R)

        
        self.I_ir0 = [Inv[i:i+self.R] for i in range(0, len(Inv), self.R)]

        # Penalty costs per location
        self.multiplier1 = float(self.params["MULTIPLIER1"])
        self.p_i = (self.df_V["P_i"] * self.multiplier1).tolist()
        
        if travel_time == 1:
            # Travel time in time periods (travel times are only 1 time period)
            self.t_ijt = np.ones((self.V, self.V), dtype=int)  # fill with 1s
            np.fill_diagonal(self.t_ijt, 0)  # set diagonal to 0
            # Loop through durationMatrix to find off-diagonal zeros
            for i in range(self.V):
                for j in range(self.V):
                    if i != j and self.durationMatrix[i][j] == 0:
                        self.t_ijt[i][j] = 0

        elif travel_time == 2:
            # Load fixed t_ijt travel time matrix from file (travel times are more than 1 time period)
            # Formulation for normalizing: t_ij = Ceiling((d_ij/d_max)*T_max).  We supposed T_max=3
            self.t_ijt = pd.read_csv(f"Data/travel_time_matrix_{self.instance_version}.csv").values.tolist()

        #Starting number of trucks at location i (time period 0)
        self.C_i0 = self.df_V["TruckStart"].tolist()
        
        # Truck delay per location
        self.h_i = self.df_V['TruckDelay'].tolist()

        # Travel cost between locations
        self.multiplier2 = float(self.params["MULTIPLIER2"])  # Scaling factor
        self.m_ij = [[(self.distanceMatrix[i][j] * self.fuel) * self.multiplier2 for j in range(len(self.df_V))] for i in range(len(self.df_V))]

        # Procurement cost per resource
        self.cost_ir = []
        idx = self.df_V[self.df_V['Type'].isin(['Warehouse', 'Supplier'])].index
        for i in range(len(self.df_V)):
            if i in idx:
                water_pc = float(self.df_V.at[i, 'Water PC'])
                food_pc = float(self.df_V.at[i, 'Food PC'])
                self.cost_ir.append([water_pc, food_pc, 0, 0])
                # self.cost_ir.append([water_pc])

        # Fraction of a truck 1 pallet of resource r requires
        self.w_r = [1 / pallet for pallet in self.df_R['Pallet Units on Truck']]
        # print("w_r:", self.w_r)
  
        # Factor of pallet required per commodity r
        self.g_shelter = self.df_R['pallet per S-Kit'].tolist()
        self.g_pod = self.df_R['pallet per P-Kit'].tolist()
        # print("g_shelter:", self.g_shelter)
        # print("g_pod:", self.g_pod)

        # Initialize K_ijt
        self.multiplier3 = float(self.params["MULTIPLIER3"])  # Scaling factor
        self.K_ijt = {}
        for index, row in self.df_K.iterrows():
            i, j, t = row['From_Node'], row['To_Node'], row['Time_Period']
            capacity = row['Capacity'] * self.multiplier3
            self.K_ijt.setdefault(i, {}).setdefault(j, {})[t] = capacity
