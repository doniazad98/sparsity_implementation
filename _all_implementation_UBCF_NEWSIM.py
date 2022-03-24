# -----------------  begin imports----------------------------------------------
import sys
import time
from ctypes import Union
from math import sqrt
import pickle
import concurrent.futures
from typing import Any
import numpy as np
from numpy import dot
from numpy.linalg import norm

# -----------------  end imports----------------------------------------------

"""
rated_items
this function returns the list of rated movies and their ratings by a target user
"""


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


"""
load_eval_data
this function loads data elavuation for each fold 
for example for fold 0 , evaluation data is in : testtrain\Movielens100k\\fold0
"""


def load_eval_data(data_name, fold):  # charger les données de test de leurs emplacement ( cas de movielens)
    # test_set, train, pairs, actual
    filename = "testtrain\Movielens100k\\fold" + str(fold) + "\\" + data_name + str(fold)
    infile = open(filename, 'rb')
    file_load = pickle.load(infile)
    infile.close()
    return (file_load)


# ---------------------------------------------------------------
def mae_1(Actual, pred):
    result = 0
    l = 0
    for i in range(0, len(Actual)):
        if pred[i] != 0.0:
            result += abs((Actual[i]) - (pred[i]))
            l += 1

    result1 = result / l
    return (result1)


def rmse_1(y_true, y_pred):  # we only consider predicted ratings
    result = 0
    l = 0
    for i in range(0, len(y_true)):
        if y_pred[i] != 0.0:
            result += ((y_true[i]) - (y_pred[i])) ** 2
            l += 1
    result1 = sqrt((result) / l)
    return (result1)


"""
compute_evaluate
this function returns the average MAE and RMSE for the 5 folds
"""


def compute_evaluate():
    result = []

    for i in range(0, 5):
        actual = load_eval_data("actual", i)
        print(len(actual))
        predicted = load_eval_data("predicted_new_sim", i)
        print(len(predicted))
        conf = confusion(actual, predicted, 3)
        cov = coverage(actual, predicted)
        prec = precision(conf[0], conf[1])
        rec = recall(conf[0], conf[3])
        f_msure = f_measure(prec, rec)
        result.append([mae_1(actual, predicted), rmse_1(actual, predicted), prec, rec, f_msure, cov])

    scores = moy_metric(result)
    print(scores)
    sys.stdout = open("results\\NEWSIM\Movielens100kresult.txt", "a+")
    print("resultas pour   avec un paramètre k =  " + str(50) + " : ")
    print("accuracy")
    print(scores)

    sys.stdout.close()
    return (result)


"""
moy_metric
this function calculates the averages from a list 
"""


def moy_metric(scores):
    mae = 0.0
    rmse = 0.0
    precision = 0.0
    recall = 0.0
    cov = 0.0
    f_mesure = 0.0
    for j in range(0, len(scores)):
        mae += scores[j][0]
        rmse += scores[j][1]
        precision += scores[j][2]
        recall += scores[j][3]
        f_mesure += scores[j][4]
        cov += scores[j][5]

    mae = mae / len(scores)
    rmse = rmse / len(scores)
    precision = precision / len(scores)
    recall = recall / len(scores)
    f_measure = f_mesure / len(scores)
    cov = cov / len(scores)
    return ([mae, rmse, precision, recall, f_measure, cov])


# ----------------- end  prepare test and train sets for 5-cross validation -------------------
def confusion(y_true, y_pred, threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    print('len du valeurs {}'.format(len(y_true)))
    for i in range(0, len(y_true)):
        # print('y_pred= {} y_true={}'.format(y_pred[i], y_true[i]))

        if y_true[i] >= threshold and y_pred[i] > threshold:
            tp += 1
        if y_true[i] >= threshold and y_pred[i] < threshold:
            fn += 1
        if y_true[i] < threshold and y_pred[i] >= threshold:
            fp += 1
        if y_true[i] < threshold and y_pred[i] <= threshold:
            tn += 1

    return [tp, fp, tn, fn]


"""
precision
this function returns precision for predicted and actual ratings
"""


def precision(tp, fp):
    result = tp / (tp + fp)
    print(result)
    return (result)


"""
recall
this function returns recall for predicted and actual ratings
"""


def recall(tp, fn):
    result = tp / (tp + fn)
    print(result)
    return (result)


"""
f_measure
this function returns f_measure for predicted and actual ratings
"""


def f_measure(precision, recall):
    result = 0
    result: Union[float, Any] = (2 * precision * recall) / (precision + recall)
    print(result)
    return result


"""
coverage
this function returns coverage for predicted and actual ratings
"""


def coverage(y_true, y_pred):
    pt = 0
    for i in range(0, len(y_true)):
        if y_pred[i] != 0.0:
            pt += 1
    result = pt / len(y_true)
    print("pt is {}".format(pt))
    return result


# ----------------- begin similarity and neighborhood selection CF   ---------------------

"""
check_neighbors_validation
this function returns (from a nearest neighbors list) users who rated the target item
"""


def check_neighbors_validation(train, movie_id):
    valid_neighbors = []
    rated = train.get(movie_id)
    if rated is not None:
        for i in range(0, len(rated)):
            if int(rated.values[i]) != 0.0:
                valid_neighbors.append(rated.index.values[i])
    else:
        valid_neighbors = []
    print(valid_neighbors)
    return valid_neighbors


"""
cosine_sim
this function returns cosine similarity for 
a target user (in test_set) and all his neighbors (in train_set)
"""


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
        sim = res / (z + 1)
        disim = (z / (z + 1)) - sim
        g = 1 / (z + 1)

    return[u2,sim,disim,g]



"""
str_list_int
this function converts a str list to int list
"""


def str_list_int(list):  # convertir une liste de STR à une list de INT
    for i in range(0, len(list)):
        list[i] = int(list[i])
    return (list)


def k_nearest_neighbors(test, train, user_id, item_id, k, distance):
    similarity = []
    neighbors = check_neighbors_validation(train, item_id)

    for i in range(0, len(neighbors)):
        user = neighbors[i]
        simi = distance(test, train, user_id, user)
        similarity.append(simi)
    print(similarity)
    sorted_list = sorted(similarity, key=lambda item: (item[3],item[1]))
    #similarity = sorted(similarity, key=lambda x: (x[1]))
    print(sorted_list[:k])
    return (sorted_list[:k])




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
    for i in range(0, len(R_user_id.index.values)):
        movie = R_user_id.index[i]
        if int(R_user_id[movie]) != 0:
            sum += int(R_user_id[movie])
            n = n + 1
    if n == 0:
        n = 1
    result = sum / n
    return (result)


"""
predict_rating_new
this function computes the predicted ratings using the weighted average formula
in case of no valid neighbors it returns 0
"""


def predict_rating_new(test, train, user_id, item_id, l, distance):
    top_res = 0
    but_res = 0
    # print("-------------------  k_valid_nearest_neighbor  ---------------------------"
    nearest_neighbors = k_nearest_neighbors(test, train, user_id, item_id, l, distance)

    if not len(nearest_neighbors):
        return 0.0

    r_target_moy = ratings_moy(test, user_id)
    for i in range(0, len(nearest_neighbors)):
        u_id = nearest_neighbors[i][0]
        s = nearest_neighbors[i][1]
        r_bar = ratings_moy(train, u_id)
        r = train.loc[u_id][item_id]
        top_res += float(s) * (float(r) - float(r_bar))
        but_res += abs(float(s))

    if but_res != 0:
        res = float(top_res) / float(but_res)
        pred = float(r_target_moy) + float(res)
    else:
        pred = 0.0

    print(pred)
    return pred


# ----------------- end prediction   ---------------------

"""
evaluate_algorithm_dataframe
This function load evaluation data for each fold, perform prediction using algorithm
and save results MAE and RMSE in txt file
"""


def evaluate_algorithm_dataframe(algorithm, distance, dataset_name, fold, *args):
    start_time = time.time()
    print("i am in evaluate")
    predicted = []

    test_set = load_eval_data("test_set", fold)
    train_set = load_eval_data("train", fold)
    pairs = load_eval_data("pairs", fold)
    actual = load_eval_data("actual", fold)

    for i in range(0, len(pairs)):
        user_id = pairs[i][0]
        item_id = pairs[i][1]
        predicted.append(algorithm(test_set, train_set, user_id, item_id, *args, distance))
    print(len(predicted))
    print(predicted)

    path = "testtrain\Movielens100k\\fold" + str(fold)
    outfile = open(path + "\\predicted_new_sim" + str(fold), 'wb')
    pickle.dump(predicted, outfile)
    outfile.close()

    scores = [mae_1(actual, predicted), rmse_1(actual, predicted)]
    sys.stdout = open("testtrain\Movielens100k\\result_new_sim_fold" + str(fold) + ".txt", "a+")
    print("resultas pour  new_sim avec un paramètre k =  " + str(*args) + " : ")
    print("accuracy")
    print(scores)
    print("That took {} seconds".format(time.time() - start_time))
    sys.stdout.close()

    return (scores)


# -----------------*******************    main    *********************--------------------------------
if __name__ == '__main__':


    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(evaluate_algorithm_dataframe(predict_rating_new, new_sim,"Movielens100k",4,50))
    
    """
    compute_evaluate()


    """
