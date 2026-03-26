import numpy as np
import demand_gen_indep
# np.random.seed(42)

test_args = {
    'dp_loc_array': [[-79.57, 33.46],
                     [-78.86, 34.04],
                     [-80.12, 33.11],
                     [-80.08, 32.93],
                     [-80.11, 32.92],
                     [-80.65, 32.95],
                     [-80.98, 32.49]
                     ],
    'track_error': np.random.normal(100, 20),
    'hurr_cat': np.random.choice([0, 1, 2, 3, 4, 5]),
    't': np.random.randint(0, demand_gen_indep.inputs.T_exteneded),
    'inputs': demand_gen_indep.inputs
}
print('Test demand mapping args', '-'*25, sep='\n')
for k, v in test_args.items():
    if k != "inputs":
        print(f'{k}: {v}')
print('Demand vector', '-'*25, demand_gen_indep.demand_generation(**test_args), sep='\n')