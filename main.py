import argparse
from input import ModelSetup
from model import OptimizationModel


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Run the optimization model for disaster logistics planning.")

    # Add arguments for input files
    parser.add_argument("-l", "--location", required=True,
                        help="Path to the location data file (e.g., 'fullDist Set V Data Location.xlsx').")
    parser.add_argument("-c", "--county", required=True,
                        help="Path to the county data file (e.g., 'fullDist Set C Data.xlsx').")
    parser.add_argument("-r", "--resource", required=True,
                        help="Path to the resource data file (e.g., 'fullDist R.xlsx').")
    parser.add_argument("-tm", "--time_matrix", required=True,
                        help="Path to the time matrix file (e.g., 'durationFULL.txt').")
    parser.add_argument("-dm", "--distance_matrix", required=True,
                        help="Path to the distance matrix file (e.g., 'distanceFULL.txt').")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the input data file (e.g., 'fullDistInputs.xlsx').")

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of the ModelSetup class
    model_setup = ModelSetup()

    # Load location data
    print(f"Loading location data from {args.location}")
    print(model_setup.load_location_data(args.location))

    # Load county data
    print(f"Loading county data from {args.county}")
    print(model_setup.load_county_data(args.county))

    # Load resource data
    print(f"Loading resource data from {args.resource}")
    print(model_setup.load_resource_data(args.resource))

    # Load time matrix
    print(f"Loading time matrix from {args.time_matrix}")
    print(model_setup.load_time_matrix(args.time_matrix))

    # Load distance matrix
    print(f"Loading distance matrix from {args.distance_matrix}")
    print(model_setup.load_distance_matrix(args.distance_matrix))

    # Load input data
    print(f"Loading input data from {args.input}")
    print(model_setup.load_input_data(args.input))

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


if __name__ == "__main__":
    main()
