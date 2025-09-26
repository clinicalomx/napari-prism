"""Spatial analyses which generate metrics at a given obs level."""


# A.K
def calculate_crossk_ripley(
    x1,
    y1,
    x2,
    y2,
    radius,
    besag_correction: bool = None,
):
    """Calculate the cross K function for two sets of points."""
