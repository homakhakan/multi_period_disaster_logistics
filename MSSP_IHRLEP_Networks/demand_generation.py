import pandas as pd
import numpy as np
import commands
from data_main import Data
from shapely.geometry import Point, LineString
import math
import helper

np.random.seed(42)

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

def in_sample(t, state, S=100):
    dim = len(state)
    in_sample = np.zeros([S, data.T, dim])
    for s in range(S):
        sampled_state = state
        in_sample[s, t, :] = state
        for t_ in range(t + 1, data.T):
            a = range(len(data.state_space[t_]))
            p = list(data.pi_mssp[t_ - 1][sampled_state].values())
            sampled_index = np.random.choice(a=a, p=p,)
            sampled_state = data.state_space[t_][sampled_index]
            in_sample[s, t_, :] = sampled_state
    return in_sample

in_sample_data = in_sample(5, data.state_space[5][0], S=100)
