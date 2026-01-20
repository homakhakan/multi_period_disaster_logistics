import pandas as pd
import numpy as np
import commands
from data_main import Data
from shapely.geometry import Point, LineString
import math
import helper

# np.random.seed(42)

args = commands.get_commands()[2]
data = Data(args=args)
data.read_data()

def demand_deterministic_landfall(data, track_err, hurr_cat, t):
    """Compute demand for the given Hurricane position, category,
    and the time period in deterministic landfall case.

    hurr_pos : tuple
        (long, lat) as the position of hurricane track.
    hurr_cat: int
        Hurricane category = 0, 1, 2, 3, 4, or 5.
    t : int
        time period = 0, 1, ...,9, or 10.
    """
    DF = np.zeros([data.I])
    # First demand is realized 4 periods before landfall.
    if t >= data.T - data.tau:
        exit(0)
        dY = [0] * data.I
        dX = [0] * data.I
        for i in range(data.I):
            hurr_pos =  [data.FORE.X[t] + data.X_rotate * track_err/data.Xmiles,
                         data.FORE.Y[t] + data.Y_rotate * track_err/data.Ymiles]
            proj_point = data.study_line.interpolate(
                data.study_line.project(Point(hurr_pos))
                )
            dY[i] = math.sqrt(
                ((proj_point.x - data.DP_GIS[i, 0]) * data.Xmiles)**2 +
                ((proj_point.y - data.DP_GIS[i, 1]) * data.Ymiles)**2
                )
            dX[i] = math.sqrt(
                ((proj_point.x - hurr_pos[0]) * data.Xmiles)**2 +
                ((proj_point.y - hurr_pos[1]) * data.Ymiles)**2
                )
            if data.demand_zone.contains(Point(hurr_pos)):
                DF_by_int = helper.demandByWS(hurr_cat)
                DF_by_time = 1 - dX[i] / data.x_max
                DF_by_track = 1 - dY[i] / data.study_region_dist
                DF[i] = DF_by_track * DF_by_time * DF_by_int
    return DF.round(3).tolist()

# t: current time, state: current hurricane state at time t, S: Number of sample paths to generate starting from "state"
def in_sample(t, state, S=100):
    dim = len(state) # Dimension of the state vector (2 in our case)
    in_sample = np.zeros([S, data.T, dim]) # A 3D NumPy array to store S in-sample scenario paths of hurricane states, across all time steps T and all state dimensions.
    for s in range(S): # Loop over the number of in-sample scenario paths
        sampled_state = state # Sets the initial state for the current path
        in_sample[s, t, :] = state # At sample path s, time t, store the values of state. # Initializes the sample path with the observed state at time t, so that the forward simulation can continue from this known point.
        for t_ in range(t + 1, data.T): # Starts one step after the known state to start simulating the future until T
            a = range(len(data.state_space[t_])) # Indices of possible states at time t_
            p = list(data.pi_mssp[t_ - 1][sampled_state].values()) # List of transition probabilities from current state to all next states
            sampled_index = np.random.choice(a=a, p=p,) # Randomly selects the next state index at time t_ based on the transition probabilities p from the current sampled_state.
            sampled_state = data.state_space[t_][sampled_index] # Extract the sampled state tuple based on its index (sampled index), e.g., (23.55, 0)
            in_sample[s, t_, :] = sampled_state # Store the sampled state for time t_ in the in-sample path s
    return in_sample # Returned object is a 3D NumPy array of shape (S, T, dim) that stores S in-sample hurricane tracks, each of length T, where each state is a vector of dimension dim.

in_sample_data = in_sample(5, data.state_space[5][0], S=2)
