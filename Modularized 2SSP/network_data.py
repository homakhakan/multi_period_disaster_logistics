import pandas as pd
import numpy as np
import math

class NetworkData:
    def __init__(self, instance_version, tperiod):
   
        self.instance_version = instance_version
        self.tperiod = tperiod

        # Load file paths
        self.file_paths = self.get_file_paths()

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
    
        return {
            "location_file": f"fullDist Set V Data Location-{self.instance_version}.xlsx",
            "county_file": f"fullDist Set C Data-{self.instance_version}.xlsx",
            "resource_file": f"fullDist R-{self.instance_version}.xlsx",
            "time_matrix_file": f"durationFull-{self.instance_version}.txt",
            "distance_matrix_file": f"distanceFull-{self.instance_version}.txt",
            "inputs_file": f"fullDistInputs-{self.instance_version}.xlsx",
            "k_ijt_file": f"K_ijt_data_fixed-{self.instance_version}.xlsx",
        }

    def define_sets(self):
      
        # C = len(self.df_C)  # Number of counties
        # V = len(self.df_V)  # Number of locations
        # R = len(self.df_R)  # Number of resources
        # T = int(self.df_inputs['Hours in 6 days'][0] / self.tperiod)  # Compute time periods

        # # Subsets of Locations
        # V0, V1, V2, V3 = [], [], [], []
        # for i in range(V):
        #     location_type = self.df_V['Type'][i]
        #     index = self.df_V['Index'][i]
        #     if location_type in ['Warehouse', 'Supplier']:
        #         V0.append(index)
        #     elif location_type == 'Shelter':
        #         V1.append(index)
        #     elif location_type == 'RSA':
        #         V2.append(index)
        #     elif location_type == 'PoD':
        #         V3.append(index)

        C = int(self.df_C.shape[0])  # Convert to regular integer
        V = int(self.df_V.shape[0])  # Convert to regular integer
        R = int(self.df_R.shape[0])  # Convert to regular integer
        T = int(self.df_inputs['Hours in 6 days'][0] / self.tperiod)  # Convert to regular integer

        # Convert NumPy integers to regular Python integers for location subsets
        V0 = [int(i) for i in self.df_V[self.df_V['Type'].isin(['Warehouse', 'Supplier'])]['Index']]
        V1 = [int(i) for i in self.df_V[self.df_V['Type'] == 'Shelter']['Index']]
        V2 = [int(i) for i in self.df_V[self.df_V['Type'] == 'RSA']['Index']]
        V3 = [int(i) for i in self.df_V[self.df_V['Type'] == 'PoD']['Index']]
        

        # Time period subsets
        T_Bef = list(range(int(T / 2)))
        T_Aft = list(range(int(T / 2), T))

        A = self.create_arc_set(V, V1, V3, C)

        return {
            "C": C, "V": V, "V0": V0, "V1": V1, "V2": V2, "V3": V3,
            "R": R, "T": T, "T_Bef": T_Bef, "T_Aft": T_Aft, "A": A
        }
    
        self.C = C

    def create_arc_set(self, V, V1, V3, C):
   
        A = set()
        no_demand = []

        # Identify locations with no demand
        for i in set(V1).union(set(V3)):
            if not any(self.df_V['County'][i] == self.df_C['County'][c] and self.df_C['Shelter Total Demand'][c] > 0 for c in range(C)):
                no_demand.append(i)

        # Generate valid arcs
        for i in range(V):
            for j in range(V):
                if i != j and i not in no_demand and j not in no_demand:
                    if (i not in V1 or j not in V3) and (i not in V3 or j not in V1):
                        A.add((i, j))

        return A

    def define_parameters(self):
    
        W_k = self.df_inputs['People on Truck'][0]  # People on truck          
        tp_salary = self.df_inputs['People on Truck'][0] * self.tperiod  # Salary per time period    
        fuel = self.df_inputs['Fuel Cost Per Mile'][0] * 0.000621371  # Fuel cost per meter (converted from miles)    

        # Opening costs per location type
        open_array = [
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

        R = len(selected_columns)  # Number of resources
        I_ir0 = [Inv[i:i+R] for i in range(0, len(Inv), R)]

        # Penalty costs per location
        multiplier1 = 1  
        p_i = (self.df_V['P_i'] * multiplier1).tolist()

        # Travel time in time periods
        seconds_per_hour = 3600
        t_ijt = [[math.ceil((self.timeMatrix[i][j] / seconds_per_hour) / self.tperiod) for j in range(len(self.df_V))] for i in range(len(self.df_V))]

        # Truck delay per location
        h_i = self.df_V['TruckDelay'].tolist()

        # Travel cost between locations
        multiplier2 = 1  # Scaling factor
        m_ij = [
            [
                ((self.distanceMatrix[i][j] * fuel) + (t_ijt[i][j] * tp_salary * W_k) + (h_i[i] * tp_salary * W_k)) * multiplier2
                for j in range(len(self.df_V))
            ]
            for i in range(len(self.df_V))
        ]

        # Procurement cost per resource
        cost_ir = []
        for i in range(len(self.df_V)):
            if i in self.df_V[self.df_V['Type'].isin(['Warehouse', 'Supplier'])].index:
                cost_ir.append([0, 0, self.df_V['Water PC'][i], self.df_V['Food PC'][i], 0, 0])

        # Fraction of a truck 1 pallet of resource r requires
        w_r = [1 / pallet for pallet in self.df_R['Pallet Units on Truck']]
        
        # Factor of pallet required per commodity r
        g_shelter = self.df_R['pallet per S-Kit'].tolist()
        g_pod = self.df_R['pallet per P-Kit'].tolist()

        # Initialize K_ijt
        K_ijt = {}
        for index, row in self.df_K.iterrows():
            i, j, t, capacity = row['From_Node'], row['To_Node'], row['Time_Period'], row['Capacity']
            K_ijt.setdefault(i, {}).setdefault(j, {})[t] = capacity

        return {
            "W_k": W_k, "tp_salary": tp_salary, "fuel": fuel, "open_array": open_array,
            "I_ir0": I_ir0, "p_i": p_i, "t_ijt": t_ijt, "h_i": h_i, "m_ij": m_ij,
            "cost_ir": cost_ir, "w_r": w_r, "g_shelter": g_shelter, "g_pod": g_pod,
            "K_ijt": K_ijt
        }
