import json
import numpy as np
import pandas as pd
import time

def load_transition_dict(path):
    with open(path, "r") as f:
        return json.load(f)

def build_node_lists(transition_dict):
    nodeLists = {}
    for t in transition_dict:
        nodeLists[int(t)] = [eval(s) for s in transition_dict[t].keys()]
    return nodeLists

def MC_sample(t, current_state, transition_dict):
    """ Draw one next state given current time t and current_state """
    
    t_key     = str(t)
    state_key = str(current_state)

    probs_dict = transition_dict[t_key][state_key]
    states_str = [s for s, p in probs_dict.items() if p > 0]
    weights    = [p for s, p in probs_dict.items() if p > 0]

    states = [eval(s) for s in states_str]
    idx    = np.random.choice(len(states), p=weights)
    return states[idx]

def termination_check(iter, relative_gap, LB, start, cutviol_iter, params):
    flag = 0
    Elapsed = time.time() - start
    
    if iter > params["MAX_ITER"]:
        flag = 1
        print("max iteration is reached")

    elif Elapsed > params["TIME_LIMIT"]:
        flag = 2
        print("time limit is reached")

    elif cutviol_iter > params["CUTVIOL_MAXITER"]:
        flag = 3
        print("cut violation is reached")

    else:
        if iter > params["STALL"]:
            relative_gap = (LB[iter-1]-LB[iter-1-params["STALL"]])/max(1e-10,abs(LB[iter-1-params["STALL"]]))
            if relative_gap < params["CUTVIOL"]:
                flag = 4
                print("the LB is not making significant progress")

    return flag, Elapsed
