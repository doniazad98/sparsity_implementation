import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler


import pickle

def read_users():
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                        encoding='latin-1')
    #print(users.head())
    return(users)


def read_items():
    i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
    movies = movies[['movie_id', 'title']]
    #print(movies.head())
    return(movies)


def read_ratings():
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                          encoding='latin-1')
    ratings = ratings.drop('timestamp', axis=1)
    #print(ratings.head())
    return(ratings)


# ---------------------------------------------------plusieur traitement
# ---camtasia studio ---
# niveux de sparcity

def sparsity(data):  # temchi ala dataframe
    count = 0
    num_users = data.index.unique().shape[0]
    num_items = data.columns.unique().shape[0]
    for k in range(0, num_users):
        for j in range(0, num_items):
            if data.values[k][j] != 0.0:
                count = count + 1
    sparsity = 1 - (count / (num_users * num_items))

    print(sparsity * 100)
    return (sparsity)


def sparsity_list(data):  # temchi ala une liste
    num_users = len(data)
    num_items = len(data[0])
    count = 0
    for k in range(0, num_users):
        # print(k)
        for j in range(0, num_items):
            # print(j)
            if data[k][j] != 0.0:
                count = count + 1
    sparsity = 1 - (count / (num_users * num_items))
    print(sparsity * 100)
    return (sparsity)


# python , dataframe ando des methodes disponible , andek index w collen , une petit base de donnes , la liste c'est brut
def dataframe_tolist(data):  # convertit liste a une dataframe
    list_data = data.values.tolist()
    print(list_data)
    return (list_data)


def ordered_items_ratings():
    # ta une matrice , les votes mba7rin , pour ordoner les item b les nombre ta3 les votes
    item_list = []
    ratings = read_ratings()
    # print(ratings.keys())
    movies = read_items()
    users = read_users()

    movie_ratings = pd.merge(movies, ratings)
    lens = pd.merge(movie_ratings, users)
    # print(lens)
    most_rated = lens.groupby('movie_id').size().sort_values(ascending=False)[:]
    final_rated = dataframe_tolist(most_rated.to_frame().reset_index())

    for w in final_rated:
        item_list.append(w[0])

    r_matrix = ratings.pivot_table(values='rating', index='user_id', columns='movie_id')
    r_matrix_dummy = r_matrix.copy().fillna(0)
    ordered_ratings = r_matrix_dummy.reindex(columns=item_list)

    print(ordered_ratings)

    return (ordered_ratings)


def pick_subdataset(data):
    # pickle ? une bib , elle sauvgarde l'objet comme il est dans la mémoire , dans disque
    for i in range(0, len(data)):
        filename = "subset" + str(i + 1)
        print(data[i])
        sparsity(data[i])
        outfile = open(filename, 'wb')
        pickle.dump(data[i], outfile)
        outfile.close()


def load_all_pick_subdataset(nb):
    # stocker dans disque l'objet
    filename = str(nb) + "-subsets"
    print(filename)
    file_sub = []
    for i in range(1, nb + 1):
        file = "subset" + str(i)
        infile = open(filename + "/" + file, 'rb')
        file_load = pickle.load(infile)
        file_sub.append(file_load)
        infile.close()

    return (file_sub)


def divide_dataset10(data, div):
    data1 = data.iloc[:, :div]

    data2 = data.iloc[:, 3:div + 3]

    data3 = data.iloc[:, 6:div + 6]

    data4 = data.iloc[:, 11:div + 11]

    data5 = data.iloc[:, 29:div + 29]

    data6 = data.iloc[:, 50:div + 50]

    data7 = data.iloc[:, 70:div + 70]

    data8 = data.iloc[:, 129:div + 129]

    data9 = data.iloc[:, 274: div + 274]

    data10 = data.iloc[:, 711:div + 711]

    return (data1, data2, data3, data4, data5, data6, data7, data8, data9, data10)


def divide_dataset5(data, div):
    data1 = data.iloc[:, 8:div + 8]

    data2 = data.iloc[:, 16:div + 16]

    data3 = data.iloc[:, 37:div + 37]

    data4 = data.iloc[:, 70:div + 70]

    data5 = data.iloc[:, 129:div + 129]

    return (data1, data2, data3, data4, data5)


def charge_subdataset(nb):
    # charger l'objet a partir du disque a la mémoire
    file_loaded = load_all_pick_subdataset(nb)
    for i in range(nb):
        sparsity(file_loaded[i])
    print(len(file_loaded))
    return (file_loaded)


# -----------------------     main   -----------------------

# créer une marice ordonnée sparsity ascendante
data = ordered_items_ratings()
data90= data.iloc[:, 350:20 + 350]
data95= data.iloc[:, 700:20 + 700]
data98= data.iloc[:, 1000:20 + 1000]

# créer 5_subsets with their sparsity
# data_div5 = divide_dataset5(data,20)
# pick_subdataset(data_div5)
# créer 10_subsets with their sparsity
# data_div10 = divide_dataset10(data,20)
# data_div5 = divide_dataset5(data,20)
'''
print(data_div10[0].columns.name)
print(data_div10[0].index.name)
print(data_div10[0].values[0][0])

for i in range(len(data_div10)):
    sparsity(data_div10[i])
pick_subdataset(data_div5)
'''

#   charger les datasets  serialisés

# saved = charge_subdataset(5)


outfile = open("data/5-subsets/subset8" , 'wb')
pickle.dump(data98, outfile)
