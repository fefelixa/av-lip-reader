import numpy as np
from scipy.interpolate import CubicSpline


def visual_feature_interp(visual_feat, audio_feat):
    """
    Return visual features matching the number of frames of the supplied audio
    feature. The time dimension must be the first dim of each feature
    matrix; uses the cubic spline interpolation method - adapted from Matlab.

    Args:
        visual_feat: the input visual features size: (visual_num_frames, visual_feature_len)
        audio_feat: the input audio features size: (audio_num_frames, audio_feature_len)

    Returns:
        visual_feat_interp: visual features that match the number of frames of the audio feature
    """

    audio_timesteps = audio_feat.shape[0]

    # Initialize an array to store interpolated visual features
    visual_feat_interp = np.zeros((audio_timesteps, visual_feat.shape[1]))

    for feature_dim in range(visual_feat.shape[1]):
        cubicSpline = CubicSpline(
            np.arange(visual_feat.shape[0]), visual_feat[:, feature_dim]
        )
        visual_feat_interp[:, feature_dim] = cubicSpline(
            np.linspace(0, visual_feat.shape[0] - 1, audio_timesteps)
        )

    return visual_feat_interp
