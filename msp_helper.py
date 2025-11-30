import json
import numpy as np
import pandas as pd
import time
np.random.seed(42)

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

    # Convert keys to strings
    t_key     = str(t)
    state_key = str(current_state)
    # Get all possible transitions
    probs_dict = transition_dict[t_key][state_key]
    # Extract valid next states and probabilities
    states_str = [s for s, p in probs_dict.items() if p > 0]
    weights    = [p for s, p in probs_dict.items() if p > 0]
    states = [eval(s) for s in states_str] # Convert state strings to tuples
    idx    = np.random.choice(len(states), p=weights) # Randomly sample next state
    return states[idx]

def create_OSpaths(nbOS, T_Bef, k_init, transition_dict, output_file):
    """
    Create out-of-sample (OOS) paths using Markov transitions.
    """
    rr = nbOS
    cc = len(T_Bef)
    print("rr = ", rr)
    print("cc = ", cc)

    # Initialize OS_paths as object array since we store tuples
    OS_paths = np.full((rr, cc), None, dtype=object)

    # Set initial state for all sample paths
    for s in range(rr):
        OS_paths[s, 0] = k_init

    # Generate paths using MC_sample
    for s in range(rr):
        k_t = k_init
        for t in range(len(T_Bef)):
            if t > 0:
                k_t = MC_sample(t-1, k_t, transition_dict)
                OS_paths[s, t] = k_t

    # Save as CSV with tuple strings
    df = pd.DataFrame(OS_paths)
    df.to_csv(output_file, index=False)

    print(f"[INFO] Generated OOS paths saved to: {output_file}")
    print(f"Shape = {OS_paths.shape}")
    return OS_paths

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
            # print(f"Stall check: LB[iter-1] = {LB[iter-1]:.6f}, LB[iter-1-{params['STALL']}] = {LB[iter-1 - params['STALL']]:.6f}, relative_gap = {relative_gap:.6e}")
            if relative_gap < params["CUTVIOL"]:
                flag = 4
                print("the LB is not making significant progress")

    return flag, Elapsed, relative_gap
