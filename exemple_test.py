import numpy as np
import pandas as pd
from all_implementation_UBCF_Cosine import cosine_sim
#------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    data = np.array([['', 'item1', 'item2','item3', 'item4'],
                        ['User1', 4, 3, 5, 4],
                        ['User2', 5, 3, 0, 0],
                        ['User3', 4, 3, 3, 4],
                        ['User4', 2, 1, 0, 0],
                        ['User5', 4, 2, 0, 0]])

    data2 = np.array([['', 'item1', 'item2','item3', 'item4', 'item5'],
                        ['User1', 2, 2, 3, 4,4],
                        ['User2', 5, 4, 2, 2,1],
                        ['User3', 2, 2, 1, 4, 1],
                        ['User4', 5, 1, 3, 2,1],
                        ['User5', 2, 5, 1, 4, 4],
                        ['User6', 0, 5, 4, 1, 0]])
    y_pred_euc = []
    y_pred_cos = []

    ratings = pd.DataFrame(data=data[1:, 1:],
                       index=data[1:, 0],
                       columns=data[0, 1:])

    ratings2 = pd.DataFrame(data=data2[1:, 1:],
                       index=data2[1:, 0],
                       columns=data2[0, 1:])

    #ratings_moy(ratings,'User5' )

    df = ratings.iloc[[1,2]]

    print(df)

    cosine_sim(ratings, ratings, 'User1', 'User5')



