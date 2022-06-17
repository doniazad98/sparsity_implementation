import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from all_implementation_UBCF_Cosine import cosine_sim


def sparsity(data):
    count = 0
    num_users = data.index.unique().shape[0]
    num_items = data.columns.unique().shape[0]
    for k in range(0, num_users):
        for j in range(0, num_items):
            if data.values[k][j] != 0.0:
                count = count + 1
    sparsity = 1 - (count / (num_users * num_items))

    #print("sparsity={}".format(sparsity*100))
    return sparsity

def read_ratings():
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=r_cols,
                          encoding='latin-1')
    ratings = ratings.drop('timestamp', axis=1)

    r_matrix = ratings.pivot_table(values="rating", index="user_id", columns="movie_id")
    r_matrix_dummy = r_matrix.copy().fillna(0)
    #print(r_matrix_dummy.head())
    return r_matrix_dummy
def load_subdataset_file(folder_name, path_subdataset_root_name, sparsity_value):
   # folder_name = str(nb)+"-subsets"
   # print(folder_name)
   # file_sub = []
   file_name = str(path_subdataset_root_name) + "_sp_" + str(sparsity_value)
   infile = open(folder_name + "/" + file_name, 'rb')
   file_load = pickle.load(infile)
   # file_sub.append(file_load)
   infile.close()
   print('Ok subdataset was loaded...')
   return (file_load)
def load_all_pick_subdataset(nb):
    filename = str(nb)+"-subsets"
   # print(filename)
    file_sub = []
    for i in range(1,9):

        file = "subset"+str(i)
        infile = open("data/"+filename+"/"+file, 'rb')
        file_load = pickle.load(infile)
        file_sub.append(file_load)
        infile.close()

    return(file_sub)

def charge_subdataset(nb):
    file_loaded = load_all_pick_subdataset(nb)
    for i in range(nb):
        sparsity(file_loaded[i])
   # print(len(file_loaded))
    return(file_loaded)
def rated_items(self, user_id):  # retourne la liste des items notés ainsi que les notes attribuées
    # print(user)
    list_movie = []
    list_ratings = []
    list_rated = []
    users = self.loc[user_id, :]
    movies = users.index
    for i in range(0, len(movies)):
        # print(users.values[i])
        if int(users.values[i]) != 0.0:
            list_movie.append(movies[i])
            list_ratings.append(int(users.values[i]))

    list_rated = [list_movie, list_ratings]
    # print(list_rated)
    return (list_rated)
def inter_rating(test, train, u1, u2):  # calculer l'intersection entre les votes de deux utilisateurs
    co_rated = []
    v1 = []
    v2 = []
    rated1 = rated_items(test, u1)
    rated2 = rated_items(train, u2)
    for i in range(0, len(rated1[0])):
        for j in range(0, len(rated2[0])):
            if rated1[0][i] == rated2[0][j]:
                co_rated.append(rated1[0][i])
                v1.append(rated1[1][i])
                v2.append(rated2[1][j])
    inter_rating = [co_rated, v1, v2]
    return (inter_rating)
def new_sim(test, train, u1, u2):  # retourne les similarités cosinus des utilisateurs par  rapport à user_id
    res = 0.0
    similarity = []
    sim = 0
    disim = 0
    g = 0
    cor = inter_rating(test, train, u1, u2)
    co_rated1 = cor[1]
    co_rated2 = cor[2]
    z = len(cor[0])
    if cor != []:
        for n in range(0, z):
            res += ((min(int(co_rated1[n]), int(co_rated2[n]))) / (max(int(co_rated1[n]), int(co_rated2[n]))))
        sim = res / (z + 2)
        disim = (z / (z + 2)) - sim
        g = 2 / (z + 2)
    print("similarity between {} and {} is = {}".format(u1,u2,[sim, disim, g]))
    return [u2, sim, disim, g]

def read_test_train():
    fic_test = open("data/subdataset_test", "rb")
    fic_train = open("data/subdataset_train", "rb")
    get_record = pickle.Unpickler(fic_test)  #
    test = pickle.load(fic_test)
    get_record = pickle.Unpickler(fic_train)
    train=pickle.load(fic_train)
    print(test,train)
    fic_train.close()
    fic_test.close()
    return test,train
if __name__ == '__main__':
    print("----------------------test des resultats--------------------")
    data = np.array([['', 'item1', 'item2', 'item3', 'item4'],
                     ['User1', 4, 3, 5, 4],
                     ['User2', 5, 3, 0, 0],
                     ['User3', 4, 3, 3, 4],
                     ['User4', 2, 1, 0, 0],
                     ['User5', 4, 2, 0, 0]])
    id_item = ['User1', 'User2', 'User3', 'User4', 'User5']
    item1_ = []
    item2_ = []
    item3_ = []
    item4_ = []
    item1 = [4, 5, 4, 2, 4]
    item2 = [3, 3, 3, 1, 2]
    item3 = [5, 0, 3, 0, 0]
    item4 = [4, 0, 4, 0, 0]
    for i in item1:
        item1_.append(i)
    for i in item2:
        item2_.append(i)
    for i in item3:
        item3_.append(i)
    for i in item4:
        item4_.append(i)


    thevector = {'item1': item1_, 'item2': item2_, 'item3': item3_, 'item4': item4_}
    data_fr = pd.DataFrame(thevector, index=id_item)
    print("the data frame is {}".format(data_fr))# we should work with that

    users = data_fr.index.values
    for i in users:
        for j in users:
            new_sim(data_fr,data_fr,i,j)