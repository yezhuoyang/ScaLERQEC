#An updated version of the main method
import numpy as np
import time

from scalerqec.Stratified.stratifiedScurveLER import format_with_uncertainty





class Scaler:
    """
    Use stratified sampling to estimate the logical error rate of a quantum error correction code.
    The only hyper parameters are physical error rate and time budget (in seconds).
    """

    def __init__(self, error_rate=0, time_budget=7200):
        self._error_rate = error_rate
        self._time_budget = time_budget


        self._subspace_LE_count={}
        self._subspace_sample_count={}





    def calculate_LER_from_file(self,filepath,pvalue,codedistance,figname,titlename, repeat=1):
        pass

