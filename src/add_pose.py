import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate):

    # Odometry: rotate 45°, move 2m, rotate 45°
    odometry = gtsam.Pose2(
        math.sqrt(2),
        math.sqrt(2),
        math.pi / 2
    )

    # Add odometry factor
    graph.add(
        gtsam.BetweenFactorPose2(
            X(3),
            X(4),
            odometry,
            ODOMETRY_NOISE
        )
    )

    # Insert initial guess for X(4)
    initial_estimate.insert(
        X(4),
        gtsam.Pose2(
            4 + math.sqrt(2),
            math.sqrt(2),
            math.pi / 2
        )
    )

    return graph, initial_estimate