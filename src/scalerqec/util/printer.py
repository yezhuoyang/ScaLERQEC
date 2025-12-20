import numpy as np


def format_with_uncertainty(value, std):
    """
    Format a value and its standard deviation in the form:
    1.23(\pm 0.45)\times 10^k
    """
    if value == 0:
        return f"0(+{std:.2e})"
    exponent = int(np.floor(np.log10(abs(value))))
    coeff = value / (10**exponent)
    std_coeff = std / (10**exponent)
    return f"{coeff:.2f}(+{std_coeff:.2f})*10^{exponent}"