import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

class ModelSetup:
    def __init__(self):
        self.model = None
        self.data = None
        self.results = None
        self.tperiod = 8  # time period is 4 hours long
        self.weight_cost = 1 / 100
        self.weight_penalty = 1 / 1
        self.df_V = None
        self.df_C = None
        self.df_R = None
        self.timeMatrix = None
        self.distanceMatrix = None
        self.df_inputs = None
        self.C = None
        self.V = None
        self.V0 = []
        self.V1 = []
        self.V2 = []
        self.V3 = []
        self.R = None
        self.T = None
        self.T_Bef = []
        self.T_Aft = []
        self.A = set()
        self.no_demand = []
        self.shelter_dist_matrix = None
        self.pod_dist_matrix = None
        self.params = {}

    def load_location_data(self, filepath):
        self.df_V = pd.read_excel(filepath)
        return self.df_V.head()

    def load_county_data(self, filepath):
        self.df_C = pd.read_excel(filepath)
        return self.df_C.head()

    def load_resource_data(self, filepath):
        self.df_R = pd.read_excel(filepath)
        return self.df_R.head()

    def load_time_matrix(self, filepath):
        self.timeMatrix = np.loadtxt(filepath)
        return self.timeMatrix

    def load_distance_matrix(self, filepath):
        self.distanceMatrix = np.loadtxt(filepath)
        return self.distanceMatrix

    def load_input_data(self, filepath):
        self.df_inputs = pd.read_excel(filepath)
        return self.df_inputs.head()

    def define_sets(self):
        # County Set
        self.C = len(self.df_C)
        print('All counties:', self.C)

        # Location Sets
        self.V = len(self.df_V)
        print('All locations:', self.V)

        for i in range(self.V):
            if self.df_V['Type'][i] == 'Warehouse' or self.df_V['Type'][i] == 'Supplier':
                self.V0.append(self.df_V['Index'][i])
        len_V0 = len(self.V0)
        print('Warehouse/Supplier locations:', self.V0)

        for i in range(self.V):
            if self.df_V['Type'][i] == 'Shelter':
                self.V1.append(self.df_V['Index'][i])
        len_V1 = len(self.V1)
        print('Shelter locations:', self.V1)

        for i in range(self.V):
            if self.df_V['Type'][i] == 'RSA':
                self.V2.append(self.df_V['Index'][i])
        len_V2 = len(self.V2)
        print('RSA locations:', self.V2)

        for i in range(self.V):
            if self.df_V['Type'][i] == 'PoD':
                self.V3.append(self.df_V['Index'][i])
        len_V3 = len(self.V3)
        print('PoD locations:', self.V3)

        # Resource Set
        self.R = len(self.df_R)
        print('All resources:', self.R)

        # Time period set
        self.T = int(self.df_inputs['Hours in 6 days'][0] / self.tperiod)
        print('Time periods:', self.T)

        for t in range(int(self.T / 2)):
            self.T_Bef.append(t)
        print('Time Before:', self.T_Bef)

        for t in range(int(self.T / 2), self.T):
            self.T_Aft.append(t)
        print('Time After:', self.T_Aft)

    def create_arc_set(self):
        # Determine no demand locations
        for i in set(self.V1).union(set(self.V3)):
            i_has_demand = any(self.df_V['County'][i] == self.df_C['County'][c] and self.df_C['Shelter Total Demand'][c] > 0 for c in range(self.C))
            if not i_has_demand:
                self.no_demand.append(i)
        print("No demand locations:", self.no_demand)

        # Create arc set
        for i in range(self.V):
            for j in range(self.V):
                if i != j and i not in self.no_demand and j not in self.no_demand:
                    if (i not in self.V1 or j not in self.V3) and (i not in self.V3 or j not in self.V1):
                        self.A.add((i, j))

        print("Number of arcs:", len(self.A))
        print("Number of arcs eliminated by this arc set:", (self.V * self.V) - len(self.A))
        # print("Arc set:", self.A)

    def create_s_curves(self):
        # Define the logistic function
        def logistic(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))

        shelter_dist = []  # Initialize a list to store the shelter demand for each county

        # Loop through each county
        for c in range(self.C):
            # Define parameters for the logistic function based on shelter data
            L = self.df_C['Shelter Total Demand'][c]  # Total amount to distribute
            k = self.df_inputs['S Curve k Value'][0]  # Steepness of the curve
            x0 = int(self.T / 4)  # Inflection point, halfway through the range (T/2)

            x_values = np.linspace(1, int(self.T / 2), int(self.T / 2))  # Generate T equally spaced data points
            logistic_values = logistic(x_values, L, k, x0)  # Calculate values using the logistic function
            logistic_values *= L / np.sum(logistic_values)  # Normalize values to sum up to the total amount
            shelter_dist.append(logistic_values)  # Add the logistic values to the shelter_dist list

        self.shelter_dist_matrix = np.array(shelter_dist)  # Convert the list of arrays to a CxT matrix

        # Print the shape of the resulting matrix
        print("Shape of the shelter_dist_matrix:", self.shelter_dist_matrix.shape)

        # Print the matrix
        print('Sum of first county demand for the first half time periods:', sum(self.shelter_dist_matrix[0][t] for t in range(int(self.T / 2))))
        print("shelter_dist_matrix:")
        print(self.shelter_dist_matrix)
        print('Example value at [1][3]:', self.shelter_dist_matrix[1][3])

    def plot_s_curves(self):
        plt.figure(figsize=(10, 6))
        for c in range(self.C):
            plt.plot(range(1, int(self.T / 2) + 1), self.shelter_dist_matrix[c], label=self.df_C['County'][c])

        plt.title('S-curves for Counties')
        plt.xlabel('Time')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid(True)
        plt.show()

    def create_normal_curves(self):
        pod_dist = [[] for _ in range(self.C)]  # Initialize a matrix to store the results

        for c in range(self.C):
            data_points = np.linspace(-3, 3, int(self.T / 2))  # Generate T/2 equally spaced points from the standard normal distribution
            pdf_values = norm.pdf(data_points)  # Calculate the probability density function (PDF) for each data point
            scaled_pdf_values = pdf_values * (self.df_C['PoD Total Demand'][c] / np.sum(pdf_values))  # Scale PDF so total area under curve is equal to PoD Demand

            pod_dist[c] = scaled_pdf_values.tolist()  # Save the scaled PDF values to the corresponding rows in the matrix

            total_distributed_amount = sum(scaled_pdf_values)  # Calculate the total distributed amount
            print("Total distributed amount for C =", c, ":", total_distributed_amount)

            for i, amount in enumerate(scaled_pdf_values):  # Print the distributed amounts at each data point
                print(f"Amount distributed at data point {i + 1}: {amount}")

        self.pod_dist_matrix = np.array(pod_dist)  # Convert the list of lists to a numpy array for easier manipulation if needed

        # Print the shape of the resulting matrix
        print("Shape of the pod_dist matrix:", self.pod_dist_matrix.shape)

    def plot_normal_curves(self):
        plt.figure(figsize=(10, 6))
        for c in range(self.C):
            plt.plot(range(1, int(self.T / 2) + 1), self.pod_dist_matrix[c], label=self.df_C['County'][c])

        plt.title('Normal Curves for Counties')
        plt.xlabel('Time')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid(True)
        plt.show()

    def define_parameters(self):
        self.params['W_k'] = self.df_inputs['People on Truck'][0]  # number of people on a truck
        self.params['tp_salary'] = self.params['W_k'] * self.tperiod  # salary per time period
        self.params['fuel'] = self.df_inputs['Fuel Cost Per Mile'][0] * 0.000621371  # fuel cost per meter
        self.params['open_array'] = [
            self.df_inputs['Open Cost V0'][0], self.df_inputs['Open Cost V1'][0],
            self.df_inputs['Open Cost V2'][0], self.df_inputs['Open Cost V3'][0]
        ]  # Assumed opening costs

        # Shelter Capacity at location i when i in V1 q_i[I]
        self.params['q_i'] = [self.df_V['Pallet Capacity'][i] for i in self.V1]

        # demand of shelter d_shelter[I][T][R] for i in V1
        d_shelter_c = self.shelter_dist_matrix

        county_capacity = []
        for c in range(self.C):
            county_capacity.append(sum(self.df_V['Pallet Capacity'][i] for i in self.V1 if self.df_V['County'][i] == self.df_C['County'][c]))

        d_shelter_i = []
        for i in self.V1:
            for t in range(self.T):
                if t in self.T_Bef:
                    for c in range(self.C):
                        if self.df_C['County'][c] == self.df_V['County'][i]:
                            d_shelter_i.append(d_shelter_c[c][t] * (self.df_V['Pallet Capacity'][i] / county_capacity[c]))
                elif t in self.T_Aft:
                    d_shelter_i.append(0)
        self.params['d_shelter'] = np.array(d_shelter_i).reshape(len(self.V1), self.T)
        self.params['d_shelter'] = np.nan_to_num(self.params['d_shelter'])

        # demand of pod d_pod[I][T][R] for i in V3
        d_pod_c = self.pod_dist_matrix

        d_pod_i = []
        for i in self.V3:
            for t in self.T_Bef:
                d_pod_i.append(0)
            for t in self.T_Aft:
                for c in range(self.C):
                    if self.df_V['County'][i] == self.df_C['County'][c]:
                        d_pod_i.append(d_pod_c[c][t - int(self.T / 2)])
        self.params['d_pod'] = np.array(d_pod_i).reshape(len(self.V3), -1)

        # starting inventory I_ir0[I][R]
        Inv = []
        for i in range(self.V):
            Inv.extend([self.df_V['S-Kit StartInv'][i],
                        self.df_V['P-Kit StartInv'][i],
                        self.df_V['Water StartInv'][i],
                        self.df_V['Food StartInv'][i],
                        self.df_V['Cot StartInv'][i],
                        self.df_V['Blanket StartInv'][i]])
        self.params['I_ir0'] = [Inv[i:i + self.R] for i in range(0, len(Inv), self.R)]

        # starting number of trucks at location i (time period 0) j_i0[I]
        self.params['j_i0'] = [self.df_V['TruckStart'][i] for i in range(self.V)]

        # penalty cost for location i p_i[I]
        self.params['p_i'] = [self.df_V['P_i'][i] for i in range(self.V)]

        # time periods to travel i,j t_ijt[i][j]
        t_ijt = []
        for i in range(self.V):
            row = []
            for j in range(self.V):
                t_ijt_hours = self.timeMatrix[i][j] / 3600
                t_ijt_period = math.ceil(t_ijt_hours / self.tperiod)
                row.append(t_ijt_period)
            t_ijt.append(row)
        self.params['t_ijt'] = t_ijt

        # distance to travel i,j e_ij[i][j]
        e_ij = []
        for i in range(self.V):
            row = []
            for j in range(self.V):
                e_ij_miles = self.distanceMatrix[i][j] * 0.000621371
                row.append(e_ij_miles)
            e_ij.append(row)
        self.params['e_ij'] = e_ij

        # truck delay h_i[i]
        self.params['h_i'] = [self.df_V['TruckDelay'][i] for i in range(self.V)]

        # cost to travel i,j m_ij[i][j]
        m_ij = []
        for i in range(self.V):
            row = []
            for j in range(self.V):
                m_ij_value = ((self.distanceMatrix[i][j] * self.params['fuel']) +
                              (self.params['t_ijt'][i][j] * self.params['tp_salary'] * self.params['W_k']) +
                              (self.params['h_i'][i] * self.params['tp_salary'] * self.params['W_k']))
                row.append(m_ij_value)
            m_ij.append(row)
        self.params['m_ij'] = m_ij

        # Fraction of a truck 1 pallet of resource r requires w_r[R]
        self.params['w_r'] = [1 / self.df_R['Pallet Units on Truck'][r] for r in range(self.R)]

        # Cost to procure resource r from location i cost_ir[i][r]
        cost = []
        for i in self.V0:
            cost.extend([0, 0, self.df_V['Water PC'][i], self.df_V['Food PC'][i], 0, 0])
        self.params['cost_ir'] = [cost[i:i + self.R] for i in range(0, len(cost), self.R)]

        # Factor of pallet required per commodity r of kit g_shelter[R]
        self.params['g_shelter'] = [self.df_R['pallet per S-Kit'][r] for r in range(self.R)]

        # Factor of pallet required per commodity r of kit g_pod[R]
        self.params['g_pod'] = [self.df_R['pallet per P-Kit'][r] for r in range(self.R)]
