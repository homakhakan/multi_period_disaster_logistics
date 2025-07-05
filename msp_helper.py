import json

def load_transition_dict(path):
    with open(path, "r") as f:
        return json.load(f)

def build_node_lists(transition_dict):
    nodeLists = {}
    for t in transition_dict:
        nodeLists[int(t)] = [eval(s) for s in transition_dict[t].keys()]
    return nodeLists

# Load the transition dictionary
transition_dict = load_transition_dict("Data/pi_mssp_d_ST_track_15.json")

# Build the node lists
node_lists = build_node_lists(transition_dict)

