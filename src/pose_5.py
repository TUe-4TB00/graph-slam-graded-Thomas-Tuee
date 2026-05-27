import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function
    pose_4 = initial_estimate.atPose2(X(4))

    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )

    return graph, initial_estimate


def add_landmark_measurement(graph, result, pose_5, landmark):
    # Add measurement from X(5) to chosen landmark
    landmark_point = result.atPoint2(L(landmark))

    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )

    return graph


def optimize(graph, initial_estimate):

    # Initialize optimizer
    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_estimate
    )

    # Perform optimization
    result = optimizer.optimize()

    # Print optimized result
    print(result)

    return result


def minimize_marginals(graph, initial_estimate, pose_options):

    best_pose = None
    best_landmark = None

    lowest_comparison = float("inf")
    best_sum = None

    # Try all poses
    for pose_name, pose_5 in pose_options.items():

        # Try both landmarks
        for landmark in [1, 2]:

            # Fresh copies
            current_graph = gtsam.NonlinearFactorGraph(graph)
            current_estimate = gtsam.Values(initial_estimate)

            # Add pose
            current_graph, current_estimate = add_pose(
                current_graph,
                current_estimate,
                pose_5
            )

            # Optimize
            result = optimize(current_graph, current_estimate)

            # Add landmark measurement
            current_graph = add_landmark_measurement(
                current_graph,
                result,
                pose_5,
                landmark
            )

            # Optimize again
            result = optimize(current_graph, current_estimate)

            # Calculate marginals
            marginals = gtsam.Marginals(current_graph, result)

            # Used for selecting best option
            comparison_value = marginals.marginalCovariance(
                L(landmark)
            ).sum()

            # Used for returned value
            sum_of_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            # Keep best result
            if comparison_value < lowest_comparison:

                lowest_comparison = comparison_value
                best_sum = sum_of_marginals

                best_pose = pose_name
                best_landmark = landmark

    return best_pose, best_landmark, best_sum


def minimize_errors(graph, initial_estimate, pose_options):

    best_pose = None
    best_landmark = None
    lowest_error = float("inf")

    # Try all poses
    for pose_name, pose_5 in pose_options.items():

        # Try both landmarks
        for landmark in [1, 2]:

            # Fresh copies
            current_graph = gtsam.NonlinearFactorGraph(graph)
            current_estimate = gtsam.Values(initial_estimate)

            # Add pose
            current_graph, current_estimate = add_pose(
                current_graph,
                current_estimate,
                pose_5
            )

            # Optimize
            result = optimize(current_graph, current_estimate)

            # Add landmark measurement
            current_graph = add_landmark_measurement(
                current_graph,
                result,
                pose_5,
                landmark
            )

            # Optimize again
            result = optimize(current_graph, current_estimate)

            # Compute errors
            list_of_errors = []

            for i in range(current_graph.size()):

                factor = current_graph.at(i)
                error = factor.error(result)

                list_of_errors.append(error)

            sum_of_errors = sum(list_of_errors)

            # Keep best result
            if sum_of_errors < lowest_error:

                lowest_error = sum_of_errors
                best_pose = pose_name
                best_landmark = landmark

    return best_pose, best_landmark, lowest_error