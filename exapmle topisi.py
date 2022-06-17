# -----------------  begin imports----------------------------------------------
import random
import sys
import time
from ctypes import Union
from math import sqrt
import pickle
import concurrent.futures
from typing import Any
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from data_preparation import sparsity
def topsis_example(decision_matrix,neighbors,user_id,k):

    e_positive=[]
    e_negative=[]


    print('decision_matrix= \n {}'.format(decision_matrix))
    # ------------------------------ Normalisation -----------------------------
    sim_ = 0
    disim_ = 0
    igno_ = 0
    normalise_decision_matrix = decision_matrix.copy()
    for i in neighbors:
        sim_ = sim_ + (normalise_decision_matrix.loc[i, 'sim'] * normalise_decision_matrix.loc[i, 'sim'])
        disim_ = disim_ + (normalise_decision_matrix.loc[i, 'disim'] * normalise_decision_matrix.loc[i, 'disim'])
        igno_ = igno_ + (normalise_decision_matrix.loc[i, 'igno'] * normalise_decision_matrix.loc[i, 'igno'])

    for i in neighbors:
        normalise_decision_matrix.loc[i, 'sim'] = normalise_decision_matrix.loc[i, 'sim'] / sqrt(sim_)
        normalise_decision_matrix.loc[i, 'disim'] = normalise_decision_matrix.loc[i, 'disim'] / sqrt(disim_)
        normalise_decision_matrix.loc[i, 'igno'] = normalise_decision_matrix.loc[i, 'igno'] / sqrt(igno_)
    print('normalise_decision_matrix= \n {}'.format(normalise_decision_matrix))
    # ------------------------------ Ponderation ----------------------------------
    weight = [1, 1, 1]
    ponderation_desicion_matrix = normalise_decision_matrix.copy()
    for i in neighbors:
        ponderation_desicion_matrix.loc[i, 'sim'] = ponderation_desicion_matrix.loc[i, 'sim'] * weight[0]
        ponderation_desicion_matrix.loc[i, 'disim'] = ponderation_desicion_matrix.loc[i, 'disim'] * weight[1]
        ponderation_desicion_matrix.loc[i, 'igno'] = ponderation_desicion_matrix.loc[i, 'igno'] * weight[2]
    print('ponderation_desicion_matrix= \n {}'.format(ponderation_desicion_matrix))
    # ------------------------------ A_positive & A_negative -------------------------

    a_positive = pd.DataFrame({'sim': max(ponderation_desicion_matrix.loc[:, "sim"].values),
                               'disim': min(ponderation_desicion_matrix.loc[:, "disim"].values),
                               'igno': min(ponderation_desicion_matrix.loc[:, "igno"].values)},
                              index=[user_id]
                              )

    a_negative = pd.DataFrame({'sim': min(ponderation_desicion_matrix.loc[:, "sim"].values),
                               'disim': max(ponderation_desicion_matrix.loc[:, "disim"].values),
                               'igno': max(ponderation_desicion_matrix.loc[:, "igno"].values)},
                              index=[user_id]
                              )
    print(" a_positive={} a_negative={}".format(a_positive,a_negative))

    # ------------------------------ E_positive & E_negative -------------------------

    euclidien_distance_positive = euclidean_distances(a_positive, ponderation_desicion_matrix)[0]
    euclidien_distance_nagative = euclidean_distances(a_negative, ponderation_desicion_matrix)[0]

    n = 0
    for i in neighbors:
        # print("[i,euclidien_distance_positive[n]]={}  [i,euclidien_distance_nagative[n]]={}".format([i,euclidien_distance_positive[n]],[i,euclidien_distance_nagative[n]]))
        e_positive.append([i, euclidien_distance_positive[n]])
        e_negative.append([i, euclidien_distance_nagative[n]])
        n = n + 1
    print("e_positive={}".format(e_positive))
    print("e_negative={}".format(e_negative))
    n = 0
    # sorted_list_positive = sorted(e_positive, key=lambda item: (item[1]))# we need minimum distance
    # sorted_list_negative = sorted(similarity, key=lambda item: (item[1]))

    # ---------------------------------relative closeness-------------------------------
    relative_closeness = []
    for i in neighbors:
        c = e_negative[n][1] / (e_positive[n][1] + e_negative[n][1])
        relative_closeness.append([i, c])
        n = n + 1
    print("relative_closeness={}".format(relative_closeness))
    # --------------------------------Ranked items----------------------------------
    final_neighbors = []
    sorted_list_positive = sorted(relative_closeness, key=lambda item: (-item[1]))  # on veut closeness maximal
    # print("sorted_list_positive={}".format(sorted_list_positive[:k]))
    sorted_list_positive = sorted_list_positive[:k]
    for i in sorted_list_positive:
        val = [i[0]]
        val.extend(decision_matrix.loc[i[0]].values)

        final_neighbors.append(val)

    print("final neighbors = {}".format(final_neighbors))

    return final_neighbors


# -----------------  end imports----------------------------------------------
from data_preparation import charge_subdataset
def load_eval_data(data_name, fold):  # charger les données de test de leurs emplacement ( cas de movielens)
    # test_set, train, pairs, actual
    filename = "testtrain\Movielens100k\\fold" + str(fold) + "\\" + data_name + str(fold)
    infile = open(filename, 'rb')
    file_load = pickle.load(infile)
    infile.close()
    return (file_load)
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
def str_list_int(list):  # convertir une liste de STR à une list de INT
    for i in range(0, len(list)):
        list[i] = int(list[i])
    return (list)
def read_test_train():
    fic_test = open("data/subdataset_test", "rb")
    fic_train = open("data/subdataset_train", "rb")
    get_record = pickle.Unpickler(fic_test)  #
    test = pickle.load(fic_test)
    get_record = pickle.Unpickler(fic_train)
    train = pickle.load(fic_train)
    # print(test,train)
    fic_train.close()
    fic_test.close()
    return test, train
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
def gen_true_values(test):
    pair = []
    y_true = []
    list_users = test.index.values.tolist()
    print(list_users)
    list_items = test.columns.values.tolist()
    for i in range(0, len(list_users)):
        user_id = list_users[i]
        for j in range(0, len(list_items)):
            if int(test.values[i][j]) != 0:
                item_id = list_items[j]
                pair.append([user_id, item_id])
                y_true.append(int(test.values[i][j]))

    return pair, y_true
def cross_validation_split_dataframe(dataset, n_folds):
    print("i am in split")
    dataset_s = dataset.copy()
    dataset_split = list()
    fold_size = int(len(dataset) / n_folds)
    users = dataset.index.values
    # print(users)
    my_list = list(users)
    random.shuffle(my_list)
    for i in range(n_folds):
        fold = list()
        fold = my_list[:fold_size]
        my_list = [x for x in my_list if x not in fold]
        fold[:] = [x - 1 for x in fold]
        fold_pd = dataset_s.iloc[fold]
        dataset_split.append(fold_pd)

    # print(dataset_split)
    return dataset_split


#******************************************* Metrics *******************************************
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

def confusion(y_true, y_pred, threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    print('len du valeurs {}'.format(len(y_true)))
    for i in range(0, len(y_true)):
        if y_pred[i] != 0.0:
            if y_true[i] >= threshold and y_pred[i] > threshold:
                tp += 1
            if y_true[i] >= threshold and y_pred[i] < threshold:
                fn += 1
            if y_true[i] < threshold and y_pred[i] >= threshold:
                fp += 1
            if y_true[i] < threshold and y_pred[i] <= threshold:
                tn += 1
        # print('y_pred= {} y_true={}'.format(y_pred[i], y_true[i]))
    return [tp, fp, tn, fn]

def precision(tp, fp):
    result = tp / (tp + fp)
    print(result)
    return (result)

def recall(tp, fn):
    result = tp / (tp + fn)
    print(result)
    return (result)

def f_measure(precision, recall):
    result = 0
    result: Union[float, Any] = (2 * precision * recall) / (precision + recall)
    print(result)
    return result

def coverage(y_true, y_pred):
    pt = 0
    for i in range(0, len(y_true)):
        if y_pred[i] != 0.0:
            pt += 1
    result = pt / len(y_true)
    print("pt is {}".format(pt))
    return result
#******************************************* Algorithm *******************************************

def check_neighbors_validation(train, movie_id):
    valid_neighbors = []
    rated = train.get(movie_id)
    if rated is not None:
        for i in range(0, len(rated)):
            if int(rated.values[i]) != 0.0:
                valid_neighbors.append(rated.index.values[i])
    else:
        valid_neighbors = []
    # print(valid_neighbors)
    return valid_neighbors

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
    #print(" similarity ={}".format([u2, sim, disim, g]))
    return [u2, sim, disim, g]

def k_nearest_neighbors(test, train, user_id, item_id, k, distance):
    similarity = []
    neighbors = check_neighbors_validation(train, item_id)

    for i in range(0, len(neighbors)):
        user = neighbors[i]
        simi = distance(test, train, user_id, user)
        similarity.append(simi)

    sorted_list = sorted(similarity, key=lambda item: (-item[1]))# V1

    #sorted_list = sorted(similarity, key=lambda item: (item[3],-item[1])) # V2
    #sorted_list = sorted(similarity, key=lambda item: (-(item[1] - item[2])))  # V3

    return sorted_list[:k]

def Topsis(test, train, user_id, item_id, k, distance):
    similarity = []

    e_positive = []
    e_negative = []
    sim = []
    dissim = []
    igno = []
    neighbors = check_neighbors_validation(train, item_id)
    if not len(neighbors):
        return []

    # ------------------------ calcule du similarity ---------------------
    for i in neighbors:
        # user = neighbors[i]
        simi = distance(test, train, user_id, i)
        similarity.append(simi)
    # print(neighbors)
    # ------------------------Creation decision matrix--------------------------------
    for i in similarity:
        sim.append(i[1])
        dissim.append(i[2])
        igno.append(i[3])
    decision_matrix = pd.DataFrame({'sim': sim, 'disim': dissim, 'igno': igno}, index=neighbors)
    # print('decision_matrix= \n {}'.format(decision_matrix))
    # ------------------------------ Normalisation -----------------------------
    sim_ = 0
    disim_ = 0
    igno_ = 0
    normalise_decision_matrix = decision_matrix.copy()
    for i in neighbors:
        sim_ = sim_ + (normalise_decision_matrix.loc[i, 'sim'] * normalise_decision_matrix.loc[i, 'sim'])
        disim_ = disim_ + (normalise_decision_matrix.loc[i, 'disim'] * normalise_decision_matrix.loc[i, 'disim'])
        igno_ = igno_ + (normalise_decision_matrix.loc[i, 'igno'] * normalise_decision_matrix.loc[i, 'igno'])

    for i in neighbors:
        normalise_decision_matrix.loc[i, 'sim'] = normalise_decision_matrix.loc[i, 'sim'] / sqrt(sim_)
        normalise_decision_matrix.loc[i, 'disim'] = normalise_decision_matrix.loc[i, 'disim'] / sqrt(disim_)
        normalise_decision_matrix.loc[i, 'igno'] = normalise_decision_matrix.loc[i, 'igno'] / sqrt(igno_)
    # print('normalise_decision_matrix= \n {}'.format(normalise_decision_matrix))
    # ------------------------------ Ponderation ----------------------------------
    weight = [1, 1, 1]
    ponderation_desicion_matrix = normalise_decision_matrix.copy()
    for i in neighbors:
        ponderation_desicion_matrix.loc[i, 'sim'] = ponderation_desicion_matrix.loc[i, 'sim'] * weight[0]
        ponderation_desicion_matrix.loc[i, 'disim'] = ponderation_desicion_matrix.loc[i, 'disim'] * weight[1]
        ponderation_desicion_matrix.loc[i, 'igno'] = ponderation_desicion_matrix.loc[i, 'igno'] * weight[2]
    # print('ponderation_desicion_matrix= \n {}'.format(ponderation_desicion_matrix))
    # ------------------------------ A_positive & A_negative -------------------------

    a_positive = pd.DataFrame({'sim': max(ponderation_desicion_matrix.loc[:, "sim"].values),
                               'disim': min(ponderation_desicion_matrix.loc[:, "disim"].values),
                               'igno': min(ponderation_desicion_matrix.loc[:, "igno"].values)},
                              index=[user_id]
                              )

    a_negative = pd.DataFrame({'sim': min(ponderation_desicion_matrix.loc[:, "sim"].values),
                               'disim': max(ponderation_desicion_matrix.loc[:, "disim"].values),
                               'igno': max(ponderation_desicion_matrix.loc[:, "igno"].values)},
                              index=[user_id]
                              )
    # print(" a_positive={} a_negative={}".format(a_positive,a_negative))

    # ------------------------------ E_positive & E_negative -------------------------

    euclidien_distance_positive = euclidean_distances(a_positive, ponderation_desicion_matrix)[0]
    euclidien_distance_nagative = euclidean_distances(a_negative, ponderation_desicion_matrix)[0]

    n = 0
    for i in neighbors:
        # print("[i,euclidien_distance_positive[n]]={}  [i,euclidien_distance_nagative[n]]={}".format([i,euclidien_distance_positive[n]],[i,euclidien_distance_nagative[n]]))
        e_positive.append([i, euclidien_distance_positive[n]])
        e_negative.append([i, euclidien_distance_nagative[n]])
        n = n + 1
    # print("e_positive={}".format(e_positive))
    n = 0
    # sorted_list_positive = sorted(e_positive, key=lambda item: (item[1]))# we need minimum distance
    # sorted_list_negative = sorted(similarity, key=lambda item: (item[1]))

    # ---------------------------------relative closeness-------------------------------
    relative_closeness = []
    for i in neighbors:
        c = e_negative[n][1] / (e_positive[n][1] + e_negative[n][1])
        relative_closeness.append([i, c])
        n = n + 1
    # --------------------------------Ranked items----------------------------------
    final_neighbors = []
    sorted_list_positive = sorted(relative_closeness, key=lambda item: (-item[1]))  # on veut closeness maximal
    # print("sorted_list_positive={}".format(sorted_list_positive[:k]))
    sorted_list_positive = sorted_list_positive[:k]
    for i in sorted_list_positive:
        val = [i[0]]
        val.extend(decision_matrix.loc[i[0]].values)

        final_neighbors.append(val)

    # print("final neighbors = {}".format(final_neighbors))

    return final_neighbors


def predict_rating_new(test, train, user_id, item_id, l, distance):
    top_res = 0
    but_res = 0
    # print("-------------------  k_valid_nearest_neighbor  ---------------------------"
    #nearest_neighbors = k_nearest_neighbors(test, train, user_id, item_id, l, distance)
    nearest_neighbors=Topsis(test, train, user_id, item_id, l, distance)

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

    # print(pred)
    return pred

def evaluate_algorithm_dataframe(algorithm, distance, dataset_name, fold, k):
    start_time = time.time()
    print("i am in evaluate")
    predicted = []
    results = open("data/results/full_2alpha_topsis" + str(k), 'wb')
    # ----------------Small dataset
    #fic_data = open("data/5-subsets/subset1", "rb")
    #record = pickle.Unpickler(fic_data)
    #data = pickle.load(fic_data)
    #train_set,test_set=train_test_split(data, test_size=0.2, random_state=25)
    # ----------------Big dataset
    test_set, train_set = read_test_train()
    sparsity(test_set)
    sparsity(train_set)


    print(test_set, train_set)
    pairs, actual = gen_true_values(test_set)
    users_test = test_set.index.values
    list = []
    print("len pairs={}".format(len(pairs)))
    kll = 0
    print("################################################################")
    for i in range(0, len(pairs)):
        begin = time.time()
        user_id = pairs[i][0]
        item_id = pairs[i][1]
        pre = algorithm(test_set, train_set, user_id, item_id, k, distance)
        predicted.append(pre)
        print("user={} item={} kll={} actual={} predicted={} time={}".format(user_id, item_id, kll, actual[kll], pre,time.time() - begin))
        kll = kll + 1
    mae = mae_1(actual, predicted)
    rmse = rmse_1(actual, predicted)
    tp, fp, tn, fn = confusion(predicted, actual, 3)
    rec = recall(tp, fn)
    presi = precision(tp, fp)
    f2 = f_measure(presi, rec)
    cov = coverage(actual, predicted)

    pickle.dump([predicted, actual], results)
    sys.stdout = open("data/results/results_full_2alpha_topsis " + str(k) + " .txt", "a+")
    print("resultas pour  new_sim avec un paramètre k =  " + str(k) + " : ")
    print(
        "mae={} \n rmse={} \n precision={} \n recall={} \n f_mesure={} \n covrage={}".format(mae, rmse, presi, rec, f2,
                                                                                             cov))
    print("That took {} seconds".format(time.time() - start_time))
    print("**********************************************************************************")
    pickle.dump([predicted, actual], results)
    # sys.stdout.close()
    results.close()


# -----------------*******************    main    *********************--------------------------------
if __name__ == '__main__':

    sim=[0.4,0.8,0.8,0.2,0.9]
    dissim=[0.2,0.05,0.15,0.6,0.1]
    igno=[0.4,0.15,0.05,0.2,0]
    neighbors=[1,2,3,4,5]
    decision_matrix = pd.DataFrame({'sim': sim, 'disim': dissim, 'igno': igno}, index=neighbors)
    topsis_example(decision_matrix,neighbors,6,5)


    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    executor.submit(evaluate_algorithm_dataframe(predict_rating_new, new_sim, "Movielens100k", 4, 20))



    """
    id_item = ['User1', 'User2', 'User3']
    item1_ = []
    item2_ = []
    item3_ = []
    item4_ = []
    item5_ = []
    item6_ = []

    item1 = [1, 0, 0]
    item2 = [2, 0, 0]

    item3 = [0, 1, 0]

    item4 = [0, 2, 0]
    item5 = [0, 0, 1]
    item6 = [0, 0, 2]

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
    thevector = {'item1': item1_, 'item2': item2_, 'item3': item3_, 'item4': item4_, 'item5': item5_, 'item6': item6_}
    data_fr = pd.DataFrame(thevector, index=id_item)
    print("the data frame is {}".format(data_fr))
    new_sim(data_fr, data_fr, "User2", "User3")

    fic_result = open("data/results/topsis50", "rb")
    record = pickle.Unpickler(fic_result)
    data_pre = pickle.load(fic_result)
    print(data_pre)
    #results = open("data/results/topsis50", 'wb')
    fic_data = open("data/5-subsets/subset1", "rb")
    record = pickle.Unpickler(fic_data)
    data = pickle.load(fic_data)

    print(data)
    test_set, train_set = train_test_split(data, test_size=0.2, random_state=25)

    fic_data.close()
    # test_set,train_set=read_test_train()
    pairs, actual = gen_true_values(test_set)
    print(actual)
    tp, fp, tn, fn=confusion(data_pre,actual,3)
    rec=recall(tp,fn)
    presi=precision(tp,fp)
    f2=f_measure(presi,rec)
    cov=coverage(actual,data_pre)
    print("precision={} \n recall={} \n f_mesure={} \n covrage={}".format(presi,rec,f2,cov))
"""

    """
    compute_evaluate()


    """
