import pickle

import pandas as pd
from sklearn.model_selection import train_test_split


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
    for i in range(1,nb+1):

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
def sparsity(data):
    count = 0
    num_users = data.index.unique().shape[0]
    num_items = data.columns.unique().shape[0]
    for k in range(0, num_users):
        for j in range(0, num_items):
            if data.values[k][j] != 0.0:
                count = count + 1
    sparsity = 1 - (count / (num_users * num_items))

    print("sparsity={}".format(sparsity*100))
    return(sparsity)

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
    read_test_train()