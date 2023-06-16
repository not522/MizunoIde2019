# Outlier control
c_lim = 0.6
c_lim_t = 0.4

# Error estimation
bootstrap_itr = 100
max_error = 2  # km

# Time window
timewindow_length = 300  # sec
timewindow_step = 150  # sec

# Source location
gridsearch_depth = 30  # km
min_source_depth = 10  # km
max_source_depth = 50  # km

# Search area
max_dist_station_source = 50  # km
max_source_depth_search = 500  # km

# Magnitude estimation
rho = 3000  # kg/m^3
beta = 2844  # m/s

# Detection criteria
min_pair_num = 16

# Station pair criteria
min_dist_station_pair = 0.3  # km
max_dist_station_pair = 100  # km

# Merge solutions
merge_source_dist = 0.2  # degree

# Min duration
min_duration = 10  # s
