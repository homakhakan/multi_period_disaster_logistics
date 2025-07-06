import json
import numpy as np
import pandas as pd

def load_transition_dict(path):
    with open(path, "r") as f:
        return json.load(f)

def build_node_lists(transition_dict):
    nodeLists = {}
    for t in transition_dict:
        nodeLists[int(t)] = [eval(s) for s in transition_dict[t].keys()]
    return nodeLists

def MC_sample(t, current_state, transition_dict):
    """
    Draw one next state given current time t and current_state.
    """
    t_key     = str(t)
    state_key = str(current_state)

    probs_dict = transition_dict[t_key][state_key]
    states_str = [s for s, p in probs_dict.items() if p > 0]
    weights    = [p for s, p in probs_dict.items() if p > 0]

    states = [eval(s) for s in states_str]
    idx    = np.random.choice(len(states), p=weights)
    return states[idx]

