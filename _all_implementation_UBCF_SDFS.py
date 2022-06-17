from ctypes import Union
import concurrent.futures
from numpy import dot
from numpy.linalg import norm
import sys
import time
from typing import Union, Any
from math import sqrt
import concurrent.futures
import pickle
import random
import numpy
"""
In this part i only read datasets files
"""


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

    return list_movie


def rated_items(self, user_id):  # retourne la liste des items notés ainsi que les notes attribuées
    # print(user)
    list_movie = []
    list_ratings = []
    list_rated = []
    users = self.loc[user_id, :]
    movies = users.index
    for i in range(0, len(movies)):
        # print(users.values[i])
        if numpy.isnan(users.values[i].all()):
            users.values[i] = numpy.nan_to_num(users.values[i])
        else:
            if int(users.values[i].all()) != 0.0:
                list_movie.append(movies[i])
                list_ratings.append(int(users.values[i].all()))

    list_rated = [list_movie, list_ratings]
    # print(list_rated)
    return (list_rated)


"""
article IJFS
"""
threshold_gamma = 50
threshold_beta = 3.52986
threshold_l = 2


def ijfs_sim(test, train, user_id, v):
    similarity = []
    users = train.index.values
    u_user = test.loc[user_id, :]
    rated_by_u = rated_items_only(test, user_id)
    moyen_ratings_u = ratings_moy(test, user_id)
    print("Before fuzzified ")
    fuzzified_fic = open("fuzzified_matrice0", "rb")
    get_record = pickle.Unpickler(fuzzified_fic)
    fuzzified_users = pickle.load(fuzzified_fic)
    # fuzzified_fic = open("fuzzified_matrice", "rb")
    # fuzzified_users=mat_fuzzifie(test,train,user_id)
    # pickle.dump(fuzzified_users,fuzzified_fic)
    # get_record = pickle.Unpickler(fuzzified_fic)
    # fuzzified_users = pickle.load(fuzzified_fic)

    print("After fuzzified ")
    sim = sdfs(threshold_l, test, train, v, user_id, rated_by_u, fuzzified_users, threshold_gamma, threshold_beta)
    print(" similarity IJFS = {}".format(sim))
    # print(" ___________________________________________________________________similarities IJFS = {}".format(sim))
    # sorted(sim, key=lambda x: (-x[1]))
    return sim



def ratings_moy(self, user_id):
    # print("je suis dans rating moy")
    sum = 0
    n = 0
    R_user_id = self.loc[user_id, :]
    #print(R_user_id)
    if R_user_id is not None:
        for i in range(0, len(R_user_id.index.values)):
            movie = R_user_id.index[i]
            r_user = R_user_id[movie]
            if r_user is not None:
                if int(r_user) != 0:
                    sum += int(r_user)
                    n = n + 1
            if n == 0:
                n = 1
    result = sum / n
    return (result)




def corated_items(rated_by_u, rated_by_v):
    corated = []
    for i in rated_by_u:
        for j in rated_by_v:
            if j == i:
                corated.append(j)
    return corated


def article_sparsity_3(test, train, user_id):
    # print("_______________________IJFS similarity begung now_________________________")
    similarity = []
    users = train.index.values
    u_user = test.loc[user_id, :]
    rated_by_u = rated_items_only(test, user_id)
    moyen_ratings_u = ratings_moy(test, user_id)
    print("Before fuzzified ")
    fuzzified_fic = open("fuzzified_matrice0", "rb")
    # fuzzified_users=mat_fuzzifie(test,train,user_id)
    # pickle.dump(fuzzified_users,fuzzified_fic)
    get_record = pickle.Unpickler(fuzzified_fic)
    fuzzified_users = pickle.load(fuzzified_fic)

    print("After fuzzified ")
    for v in users:
        sim = sdfs(threshold_l, test, train, v, user_id, rated_by_u, fuzzified_users, threshold_gamma, threshold_beta)
        print(" similarity IJFS = {}".format(sim))
        similarity.append(sim)
    print(" ___________________________________________________________________similarities IJFS = {}".format(
        sorted(similarity, key=lambda x: (-x[1]))))
    return similarity


def min_rating(u_ratings):
    min = 0
    for i in range(0, len(u_ratings)):
        if u_ratings[i] < min:
            min = u_ratings[i]
    return min


def max_rating(u_ratings):
    max = 0
    for i in range(0, len(u_ratings)):
        if u_ratings[i] < max:
            max = u_ratings[i]
    return max


def moy_ratings_i(vecteur_i):
    sum = 0
    cont = 0
    for item in vecteur_i:
        # print(" item rating is = {}".format(item))
        if item != 0:
            sum = sum + item
            cont = cont + 1
    if cont == 0:
        return 0
    else:
        # print(" moy rating i is = {}".format(sum / cont))
        return sum / cont


def nu_u(r, r_min, r_max, moyen_ratings_u):
    nu_u = 0
    if r >= r_min and r <= moyen_ratings_u:
        nu_u = (r - r_min) / (moyen_ratings_u - r_min)
    elif r >= moyen_ratings_u and r <= r_max:
        nu_u = (r - moyen_ratings_u) / (r_max - moyen_ratings_u)
    else:
        nu_u = 0
    print(" nu_u is = {}".format(nu_u))
    return nu_u


# beta is a threshold : the average of ratings in dataset

def beta(test, train):
    nb_ratings = 0
    cont_ratings = 0

    users_tr = train.index.values
    users_ts = test.index.values
    item_list = train.columns.values
    for u in users_tr:
        for i in item_list:
            r = train.loc[u, i]
            if r != 0:
                cont_ratings = cont_ratings + r
                nb_ratings = nb_ratings + 1
    for u in users_ts:
        for i in item_list:
            r = test.loc[u, i]
            if r != 0:
                cont_ratings = cont_ratings + r
                nb_ratings = nb_ratings + 1
    if nb_ratings != 0:
        return cont_ratings / nb_ratings
    else:
        return 0


def mat_fuzzifie(test, train):
    users = train.index.values
    users_ = test.index.values

    fuzzified_users = []
    item_list = train.columns.values
    for x in users:
        x_user = train.loc[x, :]
        rated_by_x = rated_items_only(train, x)
        moyen_ratings_x = ratings_moy(train, x)
        r_min = min_rating(rated_by_x)
        r_max = max_rating(rated_by_x)
        for i in item_list:
            r = x_user.loc[i]
            vecteur_i = train.loc[:, i]
            moy_r_i = moy_ratings_i(vecteur_i.values)
            if r != 0:
                nu_x_prime = nu_u(r, r_min, r_max, moyen_ratings_x)
            else:
                nu_x_prime = nu_u(moy_r_i, r_min, r_max, moyen_ratings_x)
                print(" Fazzufied value = {}".format([x, i, nu_x_prime]))
            fuzzified_users.append([x, i, nu_x_prime])
    for x in users_:
        x_user = test.loc[x, :]
        print("x_user is = {}".format(x_user))
        rated_by_x = rated_items_only(test, x)
        moyen_ratings_x = ratings_moy(test, x)
        r_min = min_rating(rated_by_x)
        r_max = max_rating(rated_by_x)
        for i in item_list:
            r = x_user.loc[i]
            vecteur_i = test.loc[:, i]
            moy_r_i = moy_ratings_i(vecteur_i.values)
            if r != 0:
                nu_x_prime = nu_u(r, r_min, r_max, moyen_ratings_x)
            else:
                nu_x_prime = nu_u(moy_r_i, r_min, r_max, moyen_ratings_x)
                print(" Fazzufied value = {}".format([x, i, nu_x_prime]))
            fuzzified_users.append([x, i, nu_x_prime])
    return fuzzified_users


def union_items(l1, l2):
    l3 = []
    l3.extend(l1)
    for ele in l2:
        if ele not in l3:
            l3.append(ele)
    return l3


def sdfs(threshold_l, test, train, v, u_user, rated_by_u, fuzzified_users, threshold_gamma, threshold_beta):
    rated_by_v = rated_items_only(train, v)
    item_list = train.columns.values
    uni = union_items(rated_by_v, rated_by_u)

    # making our fuzzified list of triplets
    # calculating T & S fuzzy norms and DFS similarity
    somme_t = 0
    somme_s = 0
    nu_u_prime = 0
    nu_v_prime = 0

    # l is the number of items being taken into account
    subset = random.sample(uni, threshold_l)
    for j in subset:
        for element in fuzzified_users:
            if element[0] == u_user and element[1] == j:
                nu_u_prime = element[2]
                print("nu_u_prime={}".format(nu_u_prime))
            if element[0] == v and element[1] == j:
                nu_v_prime = element[2]
                print("nu_v_prime={}".format(nu_v_prime))
        t_norm = min(nu_u_prime, nu_v_prime)
        somme_t = somme_t + t_norm
        s_norm = max(nu_u_prime, nu_v_prime)
        somme_s = somme_s + s_norm
    if somme_s == 0:
        dfs = 0
        print(" somme_t = {} somme_s={}".format(somme_t, somme_s))
    else:
        print(" DFS is = {} somme_t = {} somme_s={}".format(somme_t / somme_s, somme_t, somme_s))
        dfs = somme_t / somme_s

    # calculating weights ans SDFS similarity
    weight_sum = 0
    for j in uni:
        moy_r_i = moy_ratings_i(train.loc[:, j])
        w_item = weight(rated_by_u, rated_by_v, j, threshold_beta, moy_r_i)
        weight_sum = weight_sum + w_item
    sdfs = (min(weight_sum, threshold_gamma) / threshold_gamma) * dfs
    return [v, sdfs]


def weight(rated_by_u, rated_by_v, item_id, threshold_beta, u_i):
    w_item = 0
    corated_item = corated_items(rated_by_u, rated_by_v)
    if item_id in corated_item:
        w_item = 1
    else:
        if u_i <= threshold_beta:
            min_w = u_i
        else:
            min_w = threshold_beta
        w_item = (min_w / threshold_beta)
    # print("weight is {}".format(w_item))
    return w_item


def weight_sum(w_item, u_user, v_user):
    weight_sum = 0
    for i in range(u_user and v_user):
        weight_sum = weight_sum + w_item
    print("weight sum is {}".format(weight_sum))
    return weight_sum


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
    filename = "testtrain\\5-subsets\\" + data_name + str(fold)
    infile = open(filename, 'rb')
    file_load = pickle.load(infile)
    infile.close()
    return (file_load)

def load_eval_data_metrics(data_name, fold):  # charger les données de test de leurs emplacement ( cas de movielens)
    # test_set, train, pairs, actual
    filename = "results\\SDFS\\" + data_name + str(fold)
    infile = open(filename, 'rb')
    file_load = pickle.load(infile)
    infile.close()
    return (file_load)
# ---------------------------------------------------------------


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


def compute_evaluate(k):
    result = []
    i = 0 # i = 0, 2, 4
    actual = load_eval_data("actual", i)
    print(len(actual))
    predicted = load_eval_data_metrics(str(k)+"predicted_SDFS", i)
    print(len(predicted))
    conf = confusion(actual, predicted, 3)
    cov = coverage(actual, predicted)
    prec = precision(conf[0], conf[1])
    rec = recall(conf[0], conf[3])
    f_msure = f_measure(prec, rec)
    result.append([mae_1(actual, predicted), rmse_1(actual, predicted), prec, rec, f_msure, cov])
    scores = moy_metric(result)
    print(scores)
    sys.stdout = open("results\\SDFS\\predicted_SDFS.txt", "a+")
    print("resultas pour   avec un paramètre k =  " + str(k) + " : ")
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
    for i in range(0, len(y_pred)):
        # print('y_pred= {} y_true={}'.format(y_pred[i], y_true[i]))
        if y_pred[i] != 0:

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


"""
recall
this function returns recall for predicted and actual ratings
"""


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
    similarity = sorted(similarity, key=lambda x: (-x[1][1]))
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
    # print(R_user_id)
    if R_user_id is not None:
        for i in range(0, len(R_user_id.index.values)):
            movie = R_user_id.index[i]
            r_user = R_user_id[movie]
            if r_user is not None:
                if int(r_user) != 0:
                    sum += int(r_user)
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
    r_true = int(test.loc[user_id][item_id])  # added line
    test.loc[user_id][item_id] = 0  # added line
    r_target_moy = ratings_moy(test, user_id)  # added line
    nearest_neighbors = k_nearest_neighbors(test, train, user_id, item_id, l, distance)

    if not len(nearest_neighbors):
        return 0.0


    for i in range(0, len(nearest_neighbors)):
        u_id = nearest_neighbors[i][0]
        s = nearest_neighbors[i][1]
        r_bar = ratings_moy(train, u_id)
        r = train.loc[u_id][item_id]
        top_res += float(s[1]) * (float(r) - float(r_bar))
        but_res += abs(float(s[1]))

    if but_res != 0:
        res = float(top_res) / float(but_res)
        pred = float(r_target_moy) + float(res)
    else:
        pred = float(r_target_moy)
    test.loc[user_id][item_id] = float(r_true)
    print(pred)
    return pred


# ----------------- end prediction   ---------------------

"""
evaluate_algorithm_dataframe
This function load evaluation data for each fold, perform prediction using algorithm
and save results MAE and RMSE in txt file
"""


def evaluate_algorithm_dataframe(algorithm, distance, fold, *args):
    start_time = time.time()
    print("i am in evaluate")
    predicted = []
    test_set = load_eval_data("test_set",fold)
    train_set = load_eval_data("train", fold)
    pairs = load_eval_data("pairs", fold)
    actual = load_eval_data("actual", fold)
    kll = 0
    print("Number of pairs to predit = {}".format(len(pairs)))

    for i in range(0, len(pairs)):
        user_id = pairs[i][0]
        item_id = pairs[i][1]
        predicted.append(algorithm(test_set, train_set, user_id, item_id, *args, distance))
    print(len(predicted))
    print(predicted)

    path = "results\\SDFS\\"
    outfile = open(path + str(*args) + "predicted_SDFS" + str(fold), 'wb')
    pickle.dump(predicted, outfile)
    outfile.close()

    return (predicted)

    return (scores)


# -----------------*******************    main    *********************--------------------------------
if __name__ == '__main__':
    # compute_evaluate()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(evaluate_algorithm_dataframe(predict_rating_new, ijfs_sim, 0, 50))