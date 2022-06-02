# main

import numpy as np
import pandas as pd

# import required supporting methods from other libraries
from objective_weighting import weighting_methods as mcda_weights
from v_comet import VCOMET


def main():
    # main
    # choose year
    year = '2019'
    # choose type of dataset: absolute or relative
    dataset = 'absolute'
    file = 'dataset/RES_EU_' + year + '_' + dataset + '.csv'
    data = pd.read_csv(file)

    list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, len(data) + 1)]

    # choose weighting method
    weight_type = 'equal'
    # model hierarchization by decomposition
    modules = [[1,2,3,4,5], [6,7,8,9], [10,11,12], [13,14,15]]

    df_writer = pd.DataFrame()
    df_writer['Ai'] = list_alt_names
    scores = pd.DataFrame()
    for el, m in enumerate(modules):
        df_matrix = data.iloc[:, m]
        matrix = df_matrix.to_numpy()

        # in this problem there are only profit criteria
        criteria_types = np.ones(np.shape(matrix)[1])

        if weight_type == 'equal':
            weights = mcda_weights.equal_weighting(matrix)
        elif weight_type == 'entropy':
            weights = mcda_weights.entropy_weighting(matrix)
        elif weight_type == 'std':
            weights = mcda_weights.std_weighting(matrix)
        elif weight_type == 'CRITIC':
            weights = mcda_weights.critic_weighting(matrix)

        vcomet = VCOMET(normalization_method = None, v = 0.5)
        pref, _ = vcomet(matrix, weights, criteria_types)
        scores['P' + str(el + 1)] = pref

    #
    # outputs of modules are inputs for next module
    matrix = scores.to_numpy()
    # in this problem there are only profit criteria
    criteria_types = np.ones(np.shape(matrix)[1])

    if weight_type == 'equal':
        weights = mcda_weights.equal_weighting(matrix)
    elif weight_type == 'entropy':
        weights = mcda_weights.entropy_weighting(matrix)
    elif weight_type == 'std':
        weights = mcda_weights.std_weighting(matrix)
    elif weight_type == 'CRITIC':
        weights = mcda_weights.critic_weighting(matrix)

    pref, rank = vcomet(matrix, weights, criteria_types)

    df_writer['V-COMET pref'] = pref
    df_writer['V-COMET rank'] = rank

    df_writer = df_writer.set_index('Ai')
    df_writer.to_csv('results/VCOMET_RES_' + dataset + '_' + year + '.csv')


if __name__ == '__main__':
    main()