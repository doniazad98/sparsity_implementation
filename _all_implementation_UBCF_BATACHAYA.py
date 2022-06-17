# -----------------  begin imports----------------------------------------------
import math
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
from statistics import variance
# -----------------  end imports----------------------------------------------

# -----------------  begin imports----------------------------------------------
import math
import statistics
import sys
import time
from typing import Union, Any
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
from random import randrange
from csv import reader
from math import sqrt
import random
import pickle
import copy
import numpy as np
from scipy.spatial import distance
import os
from contextlib import redirect_stdout
import concurrent.futures


from numpy import linspace
# -----------------  end imports----------------------------------------------

# ----------------- begin read data--------------------------------------------
from sklearn.svm._libsvm import predict

"""
In this part i only read datasets files
"""
import math
import statistics


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
        return result
    return 0
def inter_rating_union(test, train, u1, u2):  # calculer l'intersection entre les votes de deux utilisateurs
    #print("hello inter rating")
    co_rated = []
    union = []
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
    union = rated1[0]
    union.extend(x for x in rated2[0] if x not in union)
    return ([inter_rating,union])


def rated_items_only(self, user_id):  # retourne la liste des items notés ainsi que les notes attribuées
    # print(user)
    list_movie = []
    list_ratings = []

    users = self.loc[user_id, :]
    movies = users.index
    for i in range(0, len(movies)):
        # print(users.values[i])
        if int(users.values[i]) != 0.0:
            list_movie.append(movies[i])
    # print(list_rated)
    return list_movie
def jaccard_sim(test, train, u1, u2):  # retourne les similarités jaccard des utilisateurs par  rapport à user_id
    #print("hello jaccard")
    res = inter_rating_union(test, train, u1, u2)
    intersection = res[0]
    union  = res[1]
    common_movies_rated = len(intersection[0])
    union_movies_rated = len(union)
    #print("intersection"+str(common_movies_rated))
    #print("union" + str(union_movies_rated))
    jaccard_similarity = common_movies_rated / union_movies_rated if union_movies_rated != 0 else 0
    #print(jaccard_similarity)
    return (jaccard_similarity)


"""
Bhattacharyya measure
this function returns  BC coefficient between a pair of rated items
"""


def max_rating_item(item_ratings):
    maxr = 0
    for i in range(0, len(item_ratings)):
        if item_ratings[i] > maxr:
            maxr = item_ratings[i]
    return maxr


def number_ratings(item_ratings):
    nb = 0
    for i in range(0, len(item_ratings)):
        if item_ratings[i] != 0:
            nb += 1
    # print("number of ratings in list = ", result)
    return nb


def countx(lst):
    list_final = []
    list_occurence = {}
    for ele in lst:
        if ele in list_occurence:
            list_occurence[ele] += 1
        else:
            list_occurence[ele] = 1
    for key, value in list_occurence.items():
        list_final.append((key, value))
    # print("List of ratings and their occurences = {}".format(list_final))
    # print(list_final[0][1])
    return list_final


def Bhattacharyya_measure(it1, it2):
    BC = 0
    maxrating = 0
    nb_it1 = number_ratings(it1)
    nb_it2 = number_ratings(it2)
    # print('nb_tot_rat1= {} nb_tot_rat2={}'.format(nb_it1, nb_it2))

    list_ratings1 = countx(it1)
    list_ratings2 = countx(it2)
    # print(list_ratings1)
    # print(list_ratings2)
    max_ratings1 = max_rating_item(it1)
    max_ratings2 = max_rating_item(it2)

    if max_ratings1 > max_ratings2:
        max_ratings = max_ratings1
    elif max_ratings2 > max_ratings1:
        maxrating = max_ratings2
    else:
        maxrating = max_ratings2
    # print('max_rating= {}'.format(maxrating))
    occ1 = 0
    occ2 = 0
    for h in range(1, int(maxrating) + 1):
        # print("h eqauls : ", h)
        for x in list_ratings1:
            if x[0] == h:
                occ1 = x[1]
                # print('occurence of h in it1 = {}'.format(x[1]))
        for x in list_ratings2:
            if x[0] == h:
                occ2 = x[1]
                # print('occurence of h in it2 = {}'.format(x[1]))
        BC = BC + (math.sqrt((occ1 / nb_it1) * (occ2 / nb_it2)))
    # print('Bhattacharyya_measure equals : ', BC)
    return BC


def bhattacharyya_sim(test, train, user_id, v):
    print("_______________________Bhattacharyya similarity begung now_________________________")
    users = train.index.values
    u_user = test.loc[user_id, :]
    moyen_ratings_u = ratings_moy(test, user_id)
    rated_by_u = rated_items_only(test, user_id)
    v_user = train.loc[v, :]
    moyen_ratings_v = ratings_moy(train, v)
    rated_by_v = rated_items_only(train, v)
    valeur_de_similarity = 0
    #print('before for')
    for i in rated_by_u:
        vecteur_i = train.loc[:, i]
        #print("i={}".format(i))
        for j in rated_by_v:
            #print("valeur de similarity={}".format(valeur_de_similarity))
            vecteur_j = train.loc[:, j]
            b = Bhattacharyya_measure(vecteur_i.values, vecteur_j.values)
            loc = local_similarity(u_user, v_user, moyen_ratings_u, moyen_ratings_v, i, j)
            #print('b={} loc={}'.format(b, loc))
            print("itemu={} itemv={} loc ={} bc={}".format(i,j,loc,b))
            valeur_de_similarity = valeur_de_similarity + b * loc
    #print("valeur de similarity={}".format(valeur_de_similarity))

    return valeur_de_similarity


def article_sparsity_2(test, train, user_id):
    print("_______________________Bhattacharyya similarity begung now_________________________")
    users = train.index.values
    u_user = test.loc[user_id, :]
    moyen_ratings_u = ratings_moy(test, user_id)
    rated_by_u = rated_items_only(test, user_id)
    similarity = []

    for v in users:
        v_user = train.loc[v, :]
        moyen_ratings_v = ratings_moy(train, v)
        rated_by_v = rated_items_only(train, v)
        valeur_de_similarity = 0
        for i in rated_by_u:
            vecteur_i = train.loc[:, i]
            for j in rated_by_v:
                vecteur_j = train.loc[:, j]
                b = Bhattacharyya_measure(vecteur_i.values, vecteur_j.values)
                loc = local_similarity(u_user, v_user, moyen_ratings_u, moyen_ratings_v, i, j)
                valeur_de_similarity = valeur_de_similarity + b * loc
        similarity.append([v, valeur_de_similarity])
        #print("similarities BC = {}".format(similarity))
    return similarity

def variance(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)

def local_similarity(u_user, v_user, moyen_ratings_u, moyen_ratings_v, itm1, itm2):
    loc = 0
    #print("u_usr={} ; v_user={}".format(u_user.values, v_user.values))

    var_u = variance(u_user.values)
    dev_u = math.sqrt(var_u)

    var_v = variance(v_user.values)
    dev_v = math.sqrt(var_v)


    #dev_u = statistics.stdev(u_user.values)
    #dev_v = statistics.stdev(v_user.values)
    #print('dev1= {} dev2={}'.format(dev_u, dev_v))

    rating1 = u_user.loc[itm1]
    rating2 = v_user.loc[itm2]
    if dev_v == 0 or dev_u == 0:
        return 0

    loc = (rating1 - moyen_ratings_u) * (rating2 - moyen_ratings_v) / (dev_u * dev_v)
    #print('local similarity = {}'.format(loc))
    return loc


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
        predicted = load_eval_data("predicted", i)
        print(len(predicted))
        conf = confusion(actual, predicted, 3)
        cov = coverage(actual, predicted)
        prec = precision(conf[0], conf[1])
        rec = recall(conf[0], conf[3])
        f_msure = f_measure(prec, rec)
        result.append([mae_1(actual, predicted), rmse_1(actual, predicted), prec, rec, f_msure, cov])

    scores = moy_metric(result)
    print(scores)
    sys.stdout = open("results\cosine\pred_new_Movielens100kresult.txt", "a+")
    print("resultas avec un paramètre k =  " + str(50) + " : ")
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
    # print('len du valeurs {}'.format(len(y_true)))
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
    # print(result)
    return (result)


def recall(tp, fn):
    result = tp / (tp + fn)
    # print(result)
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
    # print("pt is {}".format(pt))
    return result


# ----------------- begin similarity and neighborhood selection CF   ---------------------

"""
check_neighbors_validation
this function returns (from a nearest neighbors list) users who rated the target item
"""


def check_neighbors_validation(train, movie_id):
    valid_neighbors = []
    rated = []
    # rated = train.loc[:, movie_id]
    rated = train.get(movie_id)
    # print(rated)
    if rated is not None:
        for i in range(0, len(rated)):
            if int(rated.values[i]) != 0.0:
                valid_neighbors.append(rated.index.values[i])
    else:
        valid_neighbors = []
    # print(len(rated))

    # print(valid_neighbors)
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


def cosine_sim(test, train, u1, u2):  # retourne les similarités cosinus des utilisateurs par  rapport à user_id
    vec1 = str_list_int(test.loc[u1].tolist())
    vec2 = str_list_int(train.loc[u2].tolist())
    co_rated1 = inter_rating(test, train, u1, u2)[1]
    co_rated2 = inter_rating(test, train, u1, u2)[2]
    if co_rated1 == []: return 0
    cosine_similarity = dot(co_rated1, co_rated2) / (norm(vec1) * norm(vec2))
    return (cosine_similarity)


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
        sim = distance(test, train, user_id, user)
        if sim != 0:
            similarity.append([user, sim])
    similarity = sorted(similarity, key=lambda x: (-x[1]))
    print(similarity[:k])
    return (similarity[:k])


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

def predict_rating_new(test, train, user_id, item_id, l,distance):
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
    else :
        pred = 0.0
    print(pred)
    return pred

"""


def predict_rating_new(test, train, user_id, item_id, l, distance):
    top_res = 0
    but_res = 0
    # print("-------------------  k_valid_nearest_neighbor  ---------------------------"
    nearest_neighbors = k_nearest_neighbors(test, train, user_id, item_id, l, distance)

    if not len(nearest_neighbors):
        return 0.0

    r_true = int(test.loc[user_id][item_id])  # added line
    test.loc[user_id][item_id] = 0  # added line
    r_target_moy = ratings_moy(test, user_id)  # added line
    # r_target_moy = ratings_moy(test, user_id)
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

    # print(pred)
    test.loc[user_id][item_id] = float(r_true)
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
    kll = 0
    print("Number of pairs to predit = {}".format(len(pairs)))

    for i in range(0, len(pairs)):
        begin = time.time()
        user_id = pairs[i][0]
        item_id = pairs[i][1]
        pre = algorithm(test_set, train_set, user_id, item_id, *args, distance)
        predicted.append(pre)
        print("user={} item={} pair_number={} actual={} predicted={} time={}".format(user_id, item_id, kll, actual[kll],
                                                                                     pre, time.time() - begin))
        kll = kll + 1
    # print(len(predicted))
    # print(predicted)

    path = "testtrain\Movielens100k\\fold" + str(fold)
    outfile = open(path + "\\predicted_new" + str(fold), 'wb')
    pickle.dump(predicted, outfile)
    outfile.close()

    scores = [mae_1(actual, predicted), rmse_1(actual, predicted)]
    sys.stdout = open("testtrain\Movielens100k\\new_result_cosine_fold" + str(fold) + ".txt", "a+")
    print("resultas pour  " + str(distance) + "  avec un paramètre k =  " + str(*args) + " : ")
    print("accuracy")
    print(scores)
    print("That took {} seconds".format(time.time() - start_time))
    sys.stdout.close()

    return (scores)


# -----------------*******************    main    *********************--------------------------------
if __name__ == '__main__':

    # compute_evaluate()

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

    """
    id_item = ['User1', 'User2', 'User3']
    item1_ = []
    item2_ = []
    item3_ = []
    item4_ = []
    item5_ = []
    item6_ = []

    item1 = [1,0,0]
    item2 = [5,0,0]
    item3 = [0,1,0]
    item4 = [0,5,0]
    item5 = [0,0,1]
    item6 = [0,0,5]

    for i in item1:
        item1_.append(i)
    for i in item2:
        item2_.append(i)
    for i in item3:
        item3_.append(i)
    for i in item4:
        item4_.append(i)
    for i in item5:
        item5_.append(i)
    for i in item6:
        item6_.append(i)
    thevector = {'item1': item1_, 'item2': item2_, 'item3': item3_, 'item4': item4_,'item5': item5_,'item6': item6_}
"""

    data_fr = pd.DataFrame(thevector, index=id_item)
    print("the data frame is {}".format(data_fr))

    bc = bhattacharyya_sim(data_fr, data_fr, 'User1', 'User2')
    jacard = jaccard_sim(data_fr, data_fr, 'User1', 'User2')
    print("similarity {} et {} ={} \n BC= {} JACCARD={}".format('User1', 'User2', bc + jacard, bc, jacard))
    bc = bhattacharyya_sim(data_fr, data_fr, 'User2', 'User1')
    jacard = jaccard_sim(data_fr, data_fr, 'User2', 'User1')
    print("similarity {} et {} ={} \n BC= {} JACCARD={}".format('User2', 'User1', bc + jacard, bc, jacard))

# with concurrent.futures.ProcessPoolExecutor() as executor:
 #       executor.submit(evaluate_algorithm_dataframe(predict_rating_new, bhattacharyya_sim,"Movielens100k",4,50))
