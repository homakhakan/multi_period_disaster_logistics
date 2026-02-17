import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from IPython.display import HTML
from shapely.geometry import LineString, Point, MultiPoint, Polygon
import copy as copy
import seaborn as sns
import json
import os
import itertools
import scipy
from sklearn.preprocessing import QuantileTransformer
from dataclasses import dataclass
from typing import Callable
# np.random.seed()


# -------------------- load args from inputs.json --------------------------------------
with open('mc_data/inputs.json') as f:
    raw = json.load(f)
args = {k: (v['value'] if isinstance(v, dict) and 'value' in v else v) for k, v in raw.items()}

LONG_TO_MILES = args["LONG_TO_MILES"]
LAT_TO_MILES  = args["LAT_TO_MILES"]
H = args["H"]                      # hours of forecast horizon used
segments = args["segments"]        # number of sub-steps per 12h period
demand_lead_time = args["demand_lead_time"]  # hours
n_states_at_terminal = args["n_states_at_terminal"]
x_rotate = args["X_rotate"] # to rotate axis along study region
y_rotate = args["Y_rotate"] # to rotate axis along study region
study_region_dist = args["study_region_dist"]
demand_frac_by_cat = {int(k): float(v) for k, v in args["demand_frac_by_cat"].items()}

# -------------------- time scalars for demand mapping --------------
T = int(H / 12) + 1                 # number of 12h periods (incl 0)
T_exteneded = len(range((T - 1) * segments + 1)) # number of segmented periods
tau_12h = int(demand_lead_time / 12) + 1
# print("tau_12h:", tau_12h)
tau = len(range((tau_12h - 1) * segments + 1))
# print("tau:", tau)

# -------------------- load forecast_ext --------------------------------------------------
'''forecast_ext: The segmented deterministic baseline hurricane forecast track and intensity.
   Why needed: Provides the reference hurricane position and wind speed at every time step before adding stochastic forecast error.
'''
file_name = "forecast_ext_{}_{}_{}.csv".format(
    H,
    segments,
    n_states_at_terminal,
)
instance_dir = os.path.join("mc_output", f"mc_{H}_{segments}_{n_states_at_terminal}")
forecast_ext = pd.read_csv(os.path.join(instance_dir, file_name))

# -------------------- load study_line_extended -----------------------------------------------------
'''study_line: The polyline representing the coastline / study-region axis used for projection.
   Why needed: Used to project hurricane positions onto the region.
'''
file_name = "study_line_extended_{}_{}_{}.geojson".format(
    H,
    segments,
    n_states_at_terminal,
)
instance_dir = os.path.join("mc_output", f"mc_{H}_{segments}_{n_states_at_terminal}")
study_line_path = os.path.join(instance_dir, file_name)
study_line_extended = gpd.read_file(study_line_path).geometry.iloc[0]

# -------------------- load x_max (instance file) --------------------------------------------
'''x_max: The maximum along-track distance to the study region at the first demand-relevant lead time.
   Why needed: Normalizes the time-distance demand decay so demand smoothly ramps up as the storm approaches landfall.
'''
file_name = "x_max_{}_{}_{}.json".format(
    H,
    segments,
    n_states_at_terminal,
)
instance_dir = os.path.join("mc_output", f"mc_{H}_{segments}_{n_states_at_terminal}")
xmax_path = os.path.join(instance_dir, file_name)
with open(xmax_path, "r") as f:
    x_max = json.load(f)["x_max"]

# -------------------- load demand_zone ------------------------------------------------------------
file_name = "demand_zone_{}_{}_{}.geojson".format(
    H,
    segments,
    n_states_at_terminal,
)
instance_dir = os.path.join("mc_output", f"mc_{H}_{segments}_{n_states_at_terminal}")
demand_zone_path = os.path.join(instance_dir, file_name)
demand_zone = gpd.read_file(demand_zone_path).geometry.iloc[0]

# -------------------- load mc for deriving x_max and building state space and transition matrix -----------------------------
instance_dir = os.path.join("mc_output", f"mc_{H}_{segments}_{n_states_at_terminal}")
mc_path = os.path.join(instance_dir, f"Florence_combined_mc_p_matrix_{H}_{segments}_{n_states_at_terminal}.json")
with open(mc_path, "r") as f:
    mc_full = json.load(f)

# x_max = mc_full["x_max"] # extract x_max from MC file
# del mc_full["x_max"] # remove it from transition dictionary (important!)

# build pi_mssp 
pi_mssp = {}

for t_str, trans_dict in mc_full.items():
    t = int(t_str)
    pi_mssp[t] = {}

    for state_str, next_states in trans_dict.items():
        state = eval(state_str)   # "(0.0, 3)" → (0.0, 3)

        pi_mssp[t][state] = {eval(ns): prob for ns, prob in next_states.items()}

# # quick check
# print("pi_mssp built. Example:")
# t0 = list(pi_mssp.keys())[4]
# s0 = list(pi_mssp[t0].keys())[1]
# print("t =", t0)
# print("state =", s0)
# print("transitions =", pi_mssp[t0][s0])

# build state_space
T_mc = len(pi_mssp)
state_space = [None] * (T_mc + 1)

# states at time t = keys of pi_mssp[t]
for t in pi_mssp:
    state_space[t] = list(pi_mssp[t].keys())

# states at final time = next-states of last transition
last_t = max(pi_mssp.keys())
next_states = set()

for trans in pi_mssp[last_t].values():
    next_states.update(trans.keys())
state_space[last_t + 1] = list(next_states)

# print(state_space[10])

# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True) # defines a structured container object that groups all inputs needed by demand_generation into ONE object.
class DemandMapInputs:
    # data objects
    forecast_ext: pd.DataFrame          # columns: ['Longitude','Latitude', 'Intensity']
    study_line_extended: LineString
    demand_zone: Polygon

    # scalars
    study_region_dist: float
    x_rotate: float
    y_rotate: float
    long_to_miles: float
    lat_to_miles: float
    T_exteneded: int
    tau: int
    x_max: float

    # mapping
    demand_frac_by_cat: dict

inputs = DemandMapInputs(
    forecast_ext=forecast_ext,
    study_line_extended=study_line_extended,
    demand_zone=demand_zone,
    study_region_dist=study_region_dist,
    x_rotate=x_rotate,
    y_rotate=y_rotate,
    long_to_miles=LONG_TO_MILES,
    lat_to_miles=LAT_TO_MILES,
    T_exteneded=T_exteneded,
    tau=tau,
    x_max=x_max,
    demand_frac_by_cat=demand_frac_by_cat
)
# -----------------------------------------------------------------------------------------------

def demand_generation(dp_loc_array, track_error, hurr_cat, t, inputs):
    dp_loc_array = np.array(dp_loc_array)
    assert dp_loc_array.shape[1] == 2, "dp_loc_array must have shape (I, 2)"
    I = dp_loc_array.shape[0]
    demand = np.zeros(I)
    # from track-error and forecast, get the coordinates of the track point
    x = (
        inputs.forecast_ext.loc[t, 'Longitude'] +
        inputs.x_rotate * track_error / inputs.long_to_miles
        )
    y = (
        inputs.forecast_ext.loc[t, 'Latitude'] +
        inputs.y_rotate * track_error / inputs.lat_to_miles
        )
    proj_point = inputs.study_line_extended.interpolate(inputs.study_line_extended.project(Point([x, y])))
    # demand is zero if the track error is outside the study region
    # or if the lead-time greater than demand_lead_time (first-demand time)
    if (
        (not inputs.demand_zone.contains(Point([x, y]))) or
        (t < inputs.T_exteneded - inputs.tau)
        ):
        return demand
    # 1. demand by intensity
    demand_by_cat = float(inputs.demand_frac_by_cat[hurr_cat])
    # 2. demand by time (W.R.T. the forward speed towards study region)
    diff_x = np.sqrt(
        ((proj_point.x - x) * inputs.long_to_miles) ** 2 +
        ((proj_point.y - y) * inputs.lat_to_miles) ** 2
        )
    demand_by_time = max(1e-9, 1 - diff_x / inputs.x_max)
    # 3. demand by track error (W.R.T lateral distance along the study region)
    diff_y = np.zeros(I)
    for i in range(I):
        diff_y = np.sqrt(
            ((proj_point.x - dp_loc_array[i, 0]) * inputs.long_to_miles)**2 +
            ((proj_point.y - dp_loc_array[i, 1]) * inputs.lat_to_miles)**2
            )
        demand_by_track = max(1e-9, 1 - diff_y / inputs.study_region_dist)
        # combined demand
        demand[i] = demand_by_track * demand_by_time * demand_by_cat
    return demand

# -----------------------------------------------------------------------------------------------
""" state_space is a list over time, where each element is a list of all possible hurricane states at that time step.
    pi_mssp is a time-indexed dictionary of Markov transition probability tables that maps each hurricane state 
    at time t to a probability distribution over possible states at time t+1.

    Note: len(state_space) here is actually data.T that we used to have before.
    data.T: the total number of hurricane forecast time steps defined by the input horizon and time discretization.
"""
# t: current time, state: current hurricane state at time t, S: Number of sample paths to generate starting from "state"
def new_in_sample(t, state, S=100):
    dim = len(state) # Dimension of the state vector (2 in our case)
    in_sample = np.zeros([S, len(state_space), dim]) # A 3D NumPy array to store S in-sample scenario paths of hurricane states, across all time steps T and all state dimensions.
    for s in range(S): # Loop over the number of in-sample scenario paths
        sampled_state = state # Sets the initial state for the current path
        in_sample[s, t, :] = state # At sample path s, time t, store the values of state. # Initializes the sample path with the observed state at time t, so that the forward simulation can continue from this known point.
        for t_ in range(t + 1, len(state_space)): # Starts one step after the known state to start simulating the future until T
            a = range(len(state_space[t_])) # Indices of possible states at time t_
            p = list(pi_mssp[t_ - 1][sampled_state].values()) # List of transition probabilities from current state to all next states
            # Normalize so the probabilities sum to 1
            p = np.array(p)
            p = p / p.sum()
            sampled_index = np.random.choice(a=a, p=p,) # Randomly selects the next state index at time t_ based on the transition probabilities p from the current sampled_state.
            sampled_state = state_space[t_][sampled_index] # Extract the sampled state tuple based on its index (sampled index), e.g., (23.55, 0)
            in_sample[s, t_, :] = sampled_state # Store the sampled state for time t_ in the in-sample path s
    return in_sample # Returned object is a 3D NumPy array of shape (S, T, dim) that stores S in-sample hurricane tracks, each of length T, where each state is a vector of dimension dim.

in_sample_data = new_in_sample(5, state_space[5][0], S=2)