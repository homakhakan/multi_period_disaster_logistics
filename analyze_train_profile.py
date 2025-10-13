import pstats

p = pstats.Stats("profile_sddp_train.prof")
p.strip_dirs().sort_stats("time").print_stats(50)
