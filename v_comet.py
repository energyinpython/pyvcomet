# V-COMET method

from itertools import product
import numpy as np

# import required supporting methods from other libraries
from pyrepo_mcda.mcda_methods import VIKOR
from pyrepo_mcda.additions import rank_preferences
from mcda_method  import MCDA_method


class COMET(MCDA_method):
    def __init__(self):
        """
        Create the COMET method object
        """
        pass


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix using criteria `weights` and criteria `types`.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Vector with n criteria weights in subsequent rows. 
                Sum of all weights in vector must be equal to 1.
            types: ndarray
                Vector with n criteria types. Profit criteria are represented by 1 and cost by -1.
        
        Returns
        --------
            ndrarray, ndarray
                Vector containing preference value of each alternative. 
                In COMET method the best alternative has the highest preference value.

                Vector containing ranking of alternatives based on preference values sorted in descending order.

        Examples
        ----------
        >>> comet = COMET()
        >>> pref, rank = comet(matrix, weights, criteria_types)
        """
        COMET._verify_input_data(matrix, weights, types)
        return COMET._comet(self, matrix, weights, types)


    def _tfn(self, x, a, m, b):
        """
        Create triangular fuzzy numbers
        """
        if x < a or x > b:
            return 0
        elif a <= x < m:
            return (x-a) / (m-a)
        elif m < x <= b:
            return (b-x) / (b-m)
        elif x == m:
            return 1


    def _evaluate_alternatives(self, C, x, ind):
        """
        Evaluate alternatives
        """
        if ind == 0:
            return self._tfn(x, C[ind], C[ind], C[ind + 1])
        elif ind == len(C) - 1:
            return self._tfn(x, C[ind - 1], C[ind], C[ind])
        else:
            return self._tfn(x, C[ind - 1], C[ind], C[ind + 1])


    def _get_characteristic_values(self, matrix):
        """
        Get and return characteristic values
        """
        cv = np.zeros((matrix.shape[1], 3))
        for j in range(matrix.shape[1]):
            cv[j, 0] = np.min(matrix[:, j])
            cv[j, 1] = np.mean(matrix[:, j])
            cv[j, 2] = np.max(matrix[:, j])
        return cv


    @staticmethod
    def _comet(self, matrix, weights, types):
        # generate characteristic values
        cv = self._get_characteristic_values(matrix)
        # generate characteristic objects values based on the criteria values
        # generate matrix with COs using
        # cartesian product of characteristic values for all criteria
        co = product(*cv)
        co = np.array(list(co))
        # calculate vector SJ using VIKOR method
        vikor = VIKOR()
        sj = vikor(co, weights, types)

        # calculate vector P
        k = np.unique(sj).shape[0]
        p = np.zeros(sj.shape[0], dtype=float)

        for i in range(1, k):
            ind = sj == np.min(sj)
            p[ind] = (k - i) / (k - 1)
            sj[ind] = 1

        # inference and obtaining preference for alternatives
        preferences = []

        for i in range(len(matrix)):
            alt = matrix[i, :]
            W = []
            score = 0

            for i in range(len(p)):
                for j in range(len(co[i])):
                    ind = int(np.where(cv[j] == co[i][j])[0])
                    W.append(self._evaluate_alternatives(cv[j], alt[j], ind))
                score += np.product(W) * p[i]
                W = []
            preferences.append(score)
        preferences = np.asarray(preferences)

        rank = rank_preferences(preferences, reverse = True)
        return preferences, rank