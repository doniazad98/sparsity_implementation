# -----------------  begin imports----------------------------------------------
import sys
import time
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
import pickle
import copy
import os
from contextlib import redirect_stdout
import concurrent.futures
# -----------------  end imports----------------------------------------------

# ----------------- begin read data--------------------------------------------
"""
In this part i only read datasets files
"""

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
# ----------------- end read data----------------------------------------------

# ----------------- begin convert dataset to matrix--------------------------------------------
"""
In this part i convert the movielens dataset to matrix, i fill unrated entries with 0.0
"""
def movielens100k():
    ratings = read_ratings()
    r_matrix = ratings.pivot_table(values='rating', index='user_id', columns='movie_id')
    r_matrix_dummy = r_matrix.copy().fillna(0)
    #print(r_matrix_dummy)

    return(r_matrix_dummy)

# ----------------- end  convert dataset to matrix--------------------------------------------


# ----------------- begin prepare test and train sets for 5-cross validation -----------------------------------------
"""
cross_validation_split_20
this function splits the dataset into 5 parts, each of them having 20000 ratings
it returns a list of 5 dataframes

"""


def cross_validation_split_20(top_rating, dataset_name, n_folds):  # nbr est le numero du subdataset

    # top_rating = 20000   (20% du dataset movielens)

    dataset = movielens100k()  # lire movielens 100k sous forme de matrice
    print("i am in split 20")
    dataset_s = copy.copy(dataset)
    dataset_split = list()
    users = dataset.index.values
    print(users)
    my_list = list(users)
    print("my_list")
    print(my_list)
    random.shuffle(my_list)
    for i in range(0, n_folds):
        fold = list()
        count = 0
        for j in range(0, len(my_list)):
            if count < top_rating:
                usr = int(my_list[j])
                nb = rated_items(dataset, usr)
                count += len(nb[0])
                print(count)
                fold.append(usr)
        my_list = [x for x in my_list if x not in fold]
        fold[:] = [x - 1 for x in fold]
        fold_pd = dataset_s.iloc[fold]
        dataset_split.append(fold_pd)

    pick_cross_validate(dataset_split, dataset_name)
    for h in range(n_folds):
        print(len(dataset_split[h]))

    return dataset_split

"""
pick_cross_validate
this function permit to save the splitted dataset on disc
it is used in the former function
"""
def pick_cross_validate(data, dataset_name):
    filename = dataset_name+"_cross_validate"
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()

"""
rated_items
this function returns the list of rated movies and their ratings by a target user
"""
def rated_items(self,user_id): # retourne la liste des items notés ainsi que les notes attribuées
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

"""
gen_true_values
this function returns the list of valid [user, item] and their ratings from the test set
the resulting list is used in the evaluation
"""

def gen_true_values(test):
    pair=[]
    y_true = []

    list_users = test.index.values.tolist()
    #print(list_users)
    list_items = test.columns.values.tolist()
    for i in range(0, len(list_users)):
        user_id = list_users[i]
        for j in range(0, len(list_items)):
            if int(test.values[i][j]) != 0:
                item_id = list_items[j]
                pair.append([user_id,item_id])
                y_true.append(int(test.values[i][j]))
    return([pair,y_true])


"""
load_pick_cross_validate
this function loads the splitted dataset
"""
def load_pick_cross_validate(dataset_name):
    filename = dataset_name+"_cross_validate"
    infile = open(filename,'rb')
    file_load = pickle.load(infile)
    infile.close()
    return(file_load)

"""
gen_test_train
this function generate the test-train pairs for the 5-cross validation
we save for each fold (test_set,train_set,valid[user,items] for test and true ratings)
testtrain is the folder containing the saved  (test_set,train_set,valid[user,items] for test and true ratings)
"""

def gen_test_train(dataset_name): # sauvegarder train,test and pairs and true ratings du dataset  (dataset_name)
    folds = load_pick_cross_validate(dataset_name)
    pairs = []
    for i in range(0,5):
        train_set = folds.copy()
        test_set = train_set[i].copy()
        train_set.pop(i)
        train_set = pd.concat(train_set)
        id_pairs = gen_true_values(test_set)
        actual = id_pairs[1]
        pairs = id_pairs[0]
        path = "testtrain\Movielens100k\\fold"+str(i)
        outfile = open(path+"\pairs"+str(i), 'wb')
        pickle.dump(pairs, outfile)
        outfile = open(path+"\\train"+str(i), 'wb')
        pickle.dump(train_set, outfile)
        outfile = open(path+"\\test_set"+str(i), 'wb')
        pickle.dump(test_set, outfile)
        outfile = open(path+"\\actual"+str(i), 'wb')
        pickle.dump(actual, outfile)
        outfile.close()

    return()


"""
load_eval_data
this function loads data elavuation for each fold 
for example for fold 0 , evaluation data is in : testtrain\Movielens100k\\fold0
"""

def load_eval_data(data_name, fold): # charger les données de test de leurs emplacement ( cas de movielens)
    # test_set, train, pairs, actual
    filename = "testtrain\Movielens100k\\fold"+str(fold)+"\\"+ data_name +str(fold)
    infile = open(filename, 'rb')
    file_load = pickle.load(infile)
    infile.close()
    return(file_load)

"""
mae_1
this function returns MAE for predicted and actual ratings
"""
def mae_1(Actual,pred):
    result = 0
    l = 0

    for i in range(0, len(Actual)):
        if pred[i] != 0.0:
            result += abs((Actual[i])-(pred[i]))
            l+= 1
    print(l)
    result1 = int(result)/ l
    return(result1)

"""
rmse_1
this function returns RMSE for predicted and actual ratings
"""
def rmse_1(y_true, y_pred): # we only consider predicted ratings
    result = 0
    l = 0

    for i in range(0, len(y_true)):
        if y_pred[i] != 0.0:
            result += ((y_true[i])-(y_pred[i]))**2
            l += 1
    print(l)
    result1 = sqrt((result)/ l)
    return(result1)

"""
evaluate_algorithm_dataframe
This function load evaluation data for each fold, perform prediction using algorithm
and save results MAE and RMSE in txt file
"""

def evaluate_algorithm_dataframe(algorithm, distance,dataset_name, fold,*args):
    starttime = time.time()
    print("i am in evaluate")
    predicted = []

    test_set = load_eval_data("test_set", fold)
    train_set = load_eval_data("train", fold)
    pairs = load_eval_data("pairs", fold)
    actual = load_eval_data("actual", fold)

    for i in range (0,len(pairs)):
        user_id = pairs[i][0]
        item_id = pairs[i][1]
        predicted.append(algorithm(test_set,train_set,user_id, item_id, *args,distance))
    print(len(predicted))
    print(predicted)

    path = "testtrain\Movielens100k\\fold"+str(fold)
    outfile = open(path +"\\predicted"+ str(fold), 'wb')
    pickle.dump(predicted, outfile)
    outfile.close()

    scores = [mae_1(actual, predicted),rmse_1(actual, predicted)]
    sys.stdout = open("testtrain\Movielens100k\\result_fold"+str(fold)+".txt", "a+")
    print("resultas pour  "+str(distance)+"  avec un paramètre k =  "+str(*args)+" : ")
    print("accuracy")
    print(scores)
    print("That took {} seconds".format(time.time() - starttime))
    sys.stdout.close()

    return(scores)


"""
compute_evaluate
this function returns the average MAE and RMSE for the 5 folds
"""
def compute_evaluate():
    result = []

    for i in range (0,5):
        actual = load_eval_data("actual", i)
        print(len(actual))
        predicted = load_eval_data("predicted", i)
        print(len(predicted))

        result.append([mae_1(actual, predicted),rmse_1(actual, predicted)])

    scores = moy_metric(result)
    print(scores)
    sys.stdout = open("Movielens100kresult.txt", "a+")
    print("resultas pour   avec un paramètre k =  " + str(50) + " : ")
    print("accuracy")
    print(scores)

    sys.stdout.close()
    return(result)

"""
moy_metric
this function calculates the averages from a list 
"""
def moy_metric(scores):
    mae = 0.0
    rmse = 0.0
    for j in range(0,len(scores)):
        mae += scores[j][0]
        rmse += scores[j][1]
    mae = mae / len(scores)
    rmse = rmse / len(scores)
    return ([mae,rmse])
# ----------------- end  prepare test and train sets for 5-cross validation -------------------

# ----------------- begin similarity and neighborhood selection CF   ---------------------


"""
cosine_sim
this function returns cosine similarity for 
a target user (in test_set) and all his neighbors (in train_set)

"""

def cosine_sim(test,train,user_id):  # retourne les similarités cosinus des utilisateurs par  rapport à used_id

    users = train.index.values
    cos =[]
    vec1 = str_list_int(test.loc[user_id].tolist())
    for i in range(0, len(users)):
        index = users[i]
        if index != user_id:
            vec2 = str_list_int(train.loc[index].tolist())
            cosine_similarity = 1-cosine(vec1,vec2)
            cos.append([users[i], cosine_similarity])
    return(cos)

"""
str_list_int
this function converts a str list to int list
"""

def str_list_int(list):     # convertir une liste de STR à une list de INT
    for i in range(0, len(list)):
        list[i] = int(list[i])
    return(list)


"""
k_nearest_neighbors
this function returns the list of the  k nearest neighbors for a target user
1- it computes all neighbors for a target user
2- it sorts similarity in an ascending way
3- it selects k valid neighbors ( who rated the target item)
"""

def k_nearest_neighbors(test,train, user_id,item_id,k,distance):
    similarity = distance(test,train, user_id)
    similarity = sorted(similarity, key=lambda x: (x[1], x[0]))
    valid_neighbors = check_neighbors_validation(train, item_id, similarity)
    print(valid_neighbors[:k])
    return(valid_neighbors[:k])


"""
check_neighbors_validation
this function returns (from a nearest neighbors list) users who rated the target item
"""

def check_neighbors_validation(train, movie_id, nearest_neighbors):
    result = []
    for neighbor in nearest_neighbors:
        neighbor_id = neighbor[0]
        # print item
        rated = rated_items(train,neighbor_id)
        if movie_id in rated[0]:
            result.append(neighbor)
    return result

# ----------------- end  similarity and neighborhood selection CF   ---------------------


# ----------------- begin prediction   ---------------------
"""
ratings_moy
this function calculates the average rating for a user
"""

def ratings_moy(self, user_id):
    # print("je suis dans rating moy")
    sum = 0
    n = 0
    R_user_id = self.loc[user_id, :]
    if len(R_user_id) != 0:
        for i in range(0, len(R_user_id.index.values)):
            user = R_user_id.index[i]
            if int(R_user_id[user]) != 0:
                sum += int(R_user_id[user])
                n = n + 1
        if n == 0:
            n = 1
        result = sum / n
        print(result)
        return (result)
    return (0)

"""
predict_rating_new
this function computes the predicted ratings using the weighted average formula
in case of no valid neighbors it returns 0
"""
def predict_rating_new(test, train, user_id, item_id, l,distance):
    top_res = 0
    but_res = 0
    # print("-------------------  k_valid_nearest_neighbor  ---------------------------"
    #nearest_neighbors = k_nearest_neighbors(test, train, user_id, item_id, l, distance)
    nearest_neighbors = k_nearest_neighbors(test, train, user_id, item_id, l, distance)
    valid_neighbors = check_neighbors_validation(train, item_id, nearest_neighbors)
    if not len(valid_neighbors):
        return 0.0

    if int(test.loc[user_id][item_id]) != 0.0:
        r_true = int(test.loc[user_id][item_id])
        test.loc[user_id][item_id] = 0
        r_target_moy = ratings_moy(test, user_id)

        for i in range(0, len(valid_neighbors)):
            u_id = valid_neighbors[i][0]
            s = valid_neighbors[i][1]
            r_bar = ratings_moy(train, u_id)
            r = train.loc[u_id][item_id]
            top_res += float(s) * (float(r) - float(r_bar))
            but_res += float(s)

    if but_res != 0:
        res = float(top_res) / float(but_res)
        pred = float(r_target_moy) + float(res)
    else :
        pred = 0.0
    test.loc[user_id][item_id] = float(r_true)

    return pred


# ----------------- end prediction   ---------------------



# -----------------*******************    main    *********************--------------------------------
if __name__ == '__main__':
    y_pred_euc = []
    y_pred_cos = []
    y_true =[]



    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(evaluate_algorithm_dataframe(predict_rating_new, cosine_sim,"Movielens100k",4,50))


    compute_evaluate()



