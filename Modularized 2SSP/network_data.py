import pandas as pd
import numpy as np
import math

class NetworkData:
    def __init__(self, instance_version, tperiod):
   
        self.instance_version = instance_version
        self.tperiod = tperiod

        # Load file paths
        self.get_file_paths()

        # Load data files
        self.df_V = pd.read_excel(self.file_paths["location_file"])
        self.df_C = pd.read_excel(self.file_paths["county_file"])
        self.df_R = pd.read_excel(self.file_paths["resource_file"])
        self.df_inputs = pd.read_excel(self.file_paths["inputs_file"])
        self.df_K = pd.read_excel(self.file_paths["k_ijt_file"])

        # Load matrices
        self.timeMatrix = np.loadtxt(self.file_paths["time_matrix_file"])
        self.distanceMatrix = np.loadtxt(self.file_paths["distance_matrix_file"])

    def get_file_paths(self):
    
        self.file_paths = {
            "location_file": f"fullDist Set V Data Location-{self.instance_version}.xlsx",
            "county_file": f"fullDist Set C Data-{self.instance_version}.xlsx",
            "resource_file": f"fullDist R-{self.instance_version}.xlsx",
            "time_matrix_file": f"durationFull-{self.instance_version}.txt",
            "distance_matrix_file": f"distanceFull-{self.instance_version}.txt",
            "inputs_file": f"fullDistInputs-{self.instance_version}.xlsx",
            "k_ijt_file": f"K_ijt_data_fixed-{self.instance_version}.xlsx",
        }

    def define_sets(self):
      
        self.C = len(self.df_C)  # Number of counties
        self.V = len(self.df_V)  # Number of locations
        self.R = len(self.df_R)  # Number of resources
        self.T = int(self.df_inputs['Hours in 6 days'][0] / self.tperiod)  # Compute time periods

        # Subsets of Locations
        self.V0, self.V1, self.V2, self.V3 = [], [], [], []
        for i in range(self.V):
            location_type = self.df_V['Type'][i]
            index = self.df_V['Index'][i]
            if location_type in ['Warehouse', 'Supplier']:
                self.V0.append(index)
            elif location_type == 'Shelter':
                self.V1.append(index)
            elif location_type == 'RSA':
                self.V2.append(index)
            elif location_type == 'PoD':
                self.V3.append(index)      

        # Time period subsets
        self.T_Bef = list(range(int(self.T / 2)))
        self.T_Aft = list(range(int(self.T / 2), self.T))

        self.A = self.create_arc_set()

    def create_arc_set(self):
   
        self.A = set()
        no_demand = []

        # Identify locations with no demand
        for i in set(self.V1).union(set(self.V3)):
            if not any(self.df_V['County'][i] == self.df_C['County'][c] and self.df_C['Shelter Total Demand'][c] > 0 for c in range(self.C)):
                no_demand.append(i)

        # Generate valid arcs
        for i in range(self.V):
            for j in range(self.V):
                if i != j and i not in no_demand and j not in no_demand:
                    if (i not in self.V1 or j not in self.V3) and (i not in self.V3 or j not in self.V1):
                        self.A.add((i, j))

        return self.A

    def define_parameters(self):
    
        self.W_k = self.df_inputs['People on Truck'][0]  # People on truck          
        self.tp_salary = self.df_inputs['People on Truck'][0] * self.tperiod  # Salary per time period    
        self.fuel = self.df_inputs['Fuel Cost Per Mile'][0] * 0.000621371  # Fuel cost per meter (converted from miles)    

        # Opening costs per location type
        self.open_array = [
            self.df_inputs['Open Cost V0'][0],
            self.df_inputs['Open Cost V1'][0],
            self.df_inputs['Open Cost V2'][0],
            self.df_inputs['Open Cost V3'][0]
        ]

        # Initialize inventory list
        inventory_columns = {
            "original": ['S-Kit StartInv', 'P-Kit StartInv', 'Water StartInv', 'Food StartInv', 'Cot StartInv', 'Blanket StartInv'],
            "largesize": ['S-Kit StartInv', 'P-Kit StartInv', 'Water StartInv', 'Food StartInv', 'Cot StartInv', 'Blanket StartInv'],
            "smallsize": ['Water StartInv'],
            "mediumsize": ['Water StartInv', 'Blanket StartInv']
        }

        selected_columns = inventory_columns.get(self.instance_version, [])

        Inv = []
        for i in range(len(self.df_V)):
            Inv.extend([self.df_V[col][i] for col in selected_columns])

        self.R = len(selected_columns)  # Number of resources
        self.I_ir0 = [Inv[i:i+self.R] for i in range(0, len(Inv), self.R)]

        # Penalty costs per location
        self.multiplier1 = 1  
        self.p_i = (self.df_V['P_i'] * self.multiplier1).tolist()

        # Travel time in time periods
        # seconds_per_hour = 3600
        # self.t_ijt = [[math.ceil((self.timeMatrix[i][j] / seconds_per_hour) / self.tperiod) for j in range(len(self.df_V))] for i in range(len(self.df_V))]
        # df_t_ijt = pd.DataFrame(self.t_ijt)
        # df_t_ijt.to_csv("t_ijt_matrix.csv", index=False)
        # print("\nt_ijt matrix saved to t_ijt_matrix.csv")

        # Travel time in time periods (randomized to create variation)
        seconds_per_hour = 3600
        self.t_ijt = []

        for i in range(len(self.df_V)):
            row = []
            for j in range(len(self.df_V)):
                base_time = self.timeMatrix[i][j] / seconds_per_hour / self.tperiod
                if i == j:
                    travel_time = 0  # No self-loop
                else:
                    # Slight randomization: ceil to integer, then add extra delay for some
                    travel_time = math.ceil(base_time)
                    if travel_time == 1 and np.random.rand() < 0.3:  # 30% chance to add delay
                        travel_time += np.random.randint(1, 3)  # Add 1 or 2 extra time periods
                row.append(travel_time)
            self.t_ijt.append(row)

        # Save to CSV to inspect
        df_t_ijt = pd.DataFrame(self.t_ijt)
        df_t_ijt.to_csv("t_ijt_matrix_randomized.csv", index=False)
        print("\nt_ijt matrix with variation saved to t_ijt_matrix_randomized.csv")


        # Truck delay per location
        self.h_i = self.df_V['TruckDelay'].tolist()

        # Travel cost between locations
        self.multiplier2 = 1  # Scaling factor
        self.m_ij = [
            [
                ((self.distanceMatrix[i][j] * self.fuel) + (self.t_ijt[i][j] * self.tp_salary * self.W_k) + (self.h_i[i] * self.tp_salary * self.W_k)) * self.multiplier2
                for j in range(len(self.df_V))
            ]
            for i in range(len(self.df_V))
        ]

        # Procurement cost per resource
        self.cost_ir = []
        for i in range(len(self.df_V)):
            if i in self.df_V[self.df_V['Type'].isin(['Warehouse', 'Supplier'])].index:
                self.cost_ir.append([0, 0, self.df_V['Water PC'][i], self.df_V['Food PC'][i], 0, 0])

        # Fraction of a truck 1 pallet of resource r requires
        self.w_r = [1 / pallet for pallet in self.df_R['Pallet Units on Truck']]
        
        # Factor of pallet required per commodity r
        self.g_shelter = self.df_R['pallet per S-Kit'].tolist()
        self.g_pod = self.df_R['pallet per P-Kit'].tolist()

        # Initialize K_ijt
        self.multiplier3 = 1  # Scaling factor
        self.K_ijt = {}
        for index, row in self.df_K.iterrows():
            i, j, t = row['From_Node'], row['To_Node'], row['Time_Period']
            capacity = row['Capacity'] * self.multiplier3
            self.K_ijt.setdefault(i, {}).setdefault(j, {})[t] = capacity
