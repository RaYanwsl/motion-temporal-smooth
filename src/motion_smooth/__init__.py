"""Temporal smoothing for a video pose trajectory.

The built-in clip is a jittered walk. A WHAM-style npz with `joints` shaped
(T, J, 3), or `poses` plus `transl`, can be passed through the same steps.
No network weights are shipped.
"""

__version__ = "0.1.0"
