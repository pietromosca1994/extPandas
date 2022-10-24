import numpy as np

def shift(darray, k, axis=0):
    """
    Utility function to shift a numpy array
    Inputs
    ------
    darray: a numpy array
        the array to be shifted.
    k: integer
        number of shift
    axis: non-negative integer
        axis to perform shift operation
    Outputs
    -------
    shifted numpy array, fill the unknown values with nan
    """
    if k == 0:
        return darray
    elif k < 0:
        shift_array = np.roll(darray, k, axis=axis).astype(float)
        shift_array[k:] = np.nan
        return shift_array
    else:
        shift_array = np.roll(darray, k, axis=axis).astype(float)
        shift_array[:k] = np.nan
        return shift_array