from input import ModelSetup
from model import OptimizationModel
from output import *

print("This is just a test")

# Create an instance of the ModelSetup class
model_setup = ModelSetup()

# Load location data
location_filepath = 'fullDist Set V Data Location.xlsx'
print(model_setup.load_location_data(location_filepath))

# Load county data
county_filepath = 'fullDist Set C Data.xlsx'
print(model_setup.load_county_data(county_filepath))

# Load resource data
resource_filepath = 'fullDist R.xlsx'
print(model_setup.load_resource_data(resource_filepath))

# Load time matrix
time_matrix_filepath = "durationFULL.txt"
print(model_setup.load_time_matrix(time_matrix_filepath))

# Load distance matrix
distance_matrix_filepath = "distanceFULL.txt"
print(model_setup.load_distance_matrix(distance_matrix_filepath))

# Load input data
input_data_filepath = 'fullDistInputs.xlsx'
print(model_setup.load_input_data(input_data_filepath))

# Define and print sets
model_setup.define_sets()

# Create and print arc set
model_setup.create_arc_set()

# Create and print S curves
model_setup.create_s_curves()

# Plot S curves
model_setup.plot_s_curves()

# Create and print normal curves
model_setup.create_normal_curves()

# Plot normal curves
model_setup.plot_normal_curves()

# Define parameters
model_setup.define_parameters()

# Create an instance of the OptimizationModel class
optimization_model = OptimizationModel(model_setup)

# Define the model
optimization_model.define_model()

# Add variables
optimization_model.add_variables()

# Add constraints
optimization_model.add_constraints()

# Optimize the model
optimization_model.optimize()

# Get results
optimization_model.get_results()

print("Code execution completed successfully.")
