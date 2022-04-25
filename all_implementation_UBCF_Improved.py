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
from statistics import  variance
# -----------------  end imports----------------------------------------------
def sparsity(data):
    count = 0
    num_users = data.index.unique().shape[0]
    num_items = data.columns.unique().shape[0]
    for k in range(0, num_users):
        for j in range(0, num_items):
            if data.values[k][j] != 0.0:
                count = count + 1

    sparsity = 1 - (count / (num_users * num_items))

    print("sparsity={}".format(sparsity))
    return(sparsity)

def rated_items_only(data,user_id):# retourne la liste des items noté seulement
    list_movie = []
    users = data.loc[user_id, :]  # loc , iloc selectionner dans la list
    movies = users.index
    for i in range(0, len(movies)):
        if int(users.values[i]) != 0.0:
            list_movie.append(movies[i])
    return list_movie

def len_corated_items(test,train, user_id, v):
    co_rated = []
    rated_by_u = rated_items(test,user_id)
    rated_by_v = rated_items(train,v)
    for i in range(0, len(rated_by_u[0])):
        for j in range(0, len(rated_by_v[0])):
            if rated_by_u[0][i] == rated_by_v[0][j]:
                co_rated.append(rated_by_u[0][i])
    return len(co_rated)

def union(l1,l2):
    l3 = []
    l3.extend(l1)
    for ele in l2:
        if ele not in l3:
            l3.append(ele)
    return l3

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def munis_function(l1, l2):# l1 - l2
    l3=[]
    for i in l1:
        if i not in l2 :
            l3.append(i)
    return l3

def sparsity1article(test,train,user_id):
    users = train.index.values
    rated_by_u=rated_items_only(test,user_id)
    moyen_ratings_u=ratings_moy(test,user_id)
    similarity=[]

    #print(" rated_by_u : \n {} \n ".format(rated_by_u))
    u_user = test.loc[user_id,:]
    for v in users :
        v_user= train.loc[v,:]
        #print("user u is \n {} \n user v is \n {} ".format(u_user, v_user))
        rated_by_v=rated_items_only(train,v)
        #print("rated_by_v : {}".format(rated_by_v))
        moyen_ratings_v=ratings_moy(train,v)
        #print("Moyen ratings of {} is {} \n Item rated by {} : \n {} \n ".format(v,moyen_ratings_u, v,rated_by_u))
        totale_rated_uv=union(rated_by_v, rated_by_u) #union
        corated_uv=intersection(rated_by_v,rated_by_u)#intersection
        #print(" union {} ans {} is : {} \n intersection {} and {} is {}\n".format(user_id,v,totale_rated_uv,user_id,v,corated_uv))
        s1=S1(u_user,v_user,corated_uv,totale_rated_uv,rated_by_u,rated_by_v,moyen_ratings_u,moyen_ratings_v)
        #print("S1={}".format(s1))
        s2 = S2(corated_uv, rated_by_u, rated_by_v)
        s3 = S3(u_user, v_user, moyen_ratings_u - moyen_ratings_v)
        #print("S2={}".format(s2))
        #print("S3={}".format(s3))
        #print("Similarity between {} and {} = {}".format(user_id,v,s1*s2*s3))
        similarity.append([v,s1*s2*s3])
    print(" -------------------------------- getting out  sparcity aware")
    print("similarity= {}".format(similarity))
    return similarity

def improved_sim(test,train,user_id,v):
    users = train.index.values
    rated_by_u=rated_items_only(test,user_id)
    moyen_ratings_u=ratings_moy(test,user_id)
    rated_by_v=rated_items_only(train,v)
    moyen_ratings_v=ratings_moy(train,v)
    #print(" rated_by_u : \n {} \n ".format(rated_by_u))
    u_user = test.loc[user_id,:]
    v_user= train.loc[v,:]
    totale_rated_uv=union(rated_by_v, rated_by_u) #union
    corated_uv=intersection(rated_by_v,rated_by_u)#intersection
    #print("user u is \n {} \n user v is \n {} ".format(u_user, v_user))
    #print("rated_by_v : {}".format(rated_by_v))
    #print("Moyen ratings of {} is {} \n Item rated by {} : \n {} \n ".format(v,moyen_ratings_u, v,rated_by_u))
    #print(" union {} ans {} is : {} \n intersection {} and {} is {}\n".format(user_id,v,totale_rated_uv,user_id,v,corated_uv))
    s1=S1(u_user,v_user,corated_uv,totale_rated_uv,rated_by_u,rated_by_v,moyen_ratings_u,moyen_ratings_v)
    s2 = S2(corated_uv, rated_by_u, rated_by_v)
    s3 = S3(u_user, v_user, moyen_ratings_u - moyen_ratings_v)
    #print("S1={}".format(s1))
    #print("S2={}".format(s2))
    #print("S3={}".format(s3))
    #print("Similarity between {} and {} = {}".format(user_id,v,s1*s2*s3))
    return s1*s2*s3

def S1(u_user,v_user,corated_uv,totale_rated_uv,rated_by_u,rated_by_v,moyen_ratings_u,moyen_ratings_v):
    up=0
    down_u=0
    down_v=0
    down_u_moy=0
    down_v_moy=0
    if sparsity_level < sparsity_thershold:
        # only working with corated items
        for i in corated_uv:
            up=up+ (u_user.loc[i]*v_user.loc[i])
        for i in rated_by_u:
            #print("u_user={} ".format(u_user.loc[i]))
            down_u = down_u + (u_user.loc[i] * u_user.loc[i])
        for i in rated_by_v:
            #print("v_user={} ".format(v_user.loc[i]))
            down_v = down_v + (v_user.loc[i] * v_user.loc[i])
        #print(" up= {} , down_u={} , down_v={}".format(up ,down_u,down_v))
        if down_u==0 or down_v==0 :
            return 0
        else :
            return up/(math.sqrt(down_u) * math.sqrt(down_v))


    else :
        #working with all the rated item of both users
        for i in corated_uv:
            up=up+ (u_user.loc[i]*v_user.loc[i])
        j=munis_function(rated_by_u,corated_uv)
        for i in j :# formally u_moins_v
            up = up + (u_user.loc[i] * moyen_ratings_v)
        k=munis_function(rated_by_v,corated_uv)
        for i in k: # formally v_moins_u
            up = up + (v_user.loc[i] * moyen_ratings_u)
        for i in rated_by_u:
            down_u = down_u + (u_user.loc[i] * u_user.loc[i])
        for i in rated_by_v:
            down_v = down_v + (v_user.loc[i] * v_user.loc[i])
        l=munis_function(totale_rated_uv,rated_by_u)
        for i in l :
            down_u_moy=down_u_moy+(moyen_ratings_u*moyen_ratings_u)
        m=munis_function(totale_rated_uv,rated_by_v)
        for i in m :
            down_v_moy=down_v_moy+(moyen_ratings_v*moyen_ratings_v)

        return up/(math.sqrt(down_u+down_u_moy)*math.sqrt(down_v+down_v_moy))

def S2(corated_uv,rated_by_u,rated_by_v):# im sure about this function
    var1=(len(corated_uv)*len(corated_uv))/(len(rated_by_u)*len(rated_by_v))
    var2=1+math.exp(-var1)
    return 1/var2

def S3(u_user,v_user,def_moyen):
    #print("im in S3 , values are {} ".format(u_user.values))
    var_u=variance(u_user.values)
    var_v = variance(v_user.values)
    def_variance=var_u-var_v
    #print("the variances are {} and {}".format(var_u,var_v))
    var3=1+math.exp(-abs(def_variance)*abs(def_moyen))
    return 1-(1/var3)



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

# ---------------------------------------------------------------
def mae_1(Actual,pred):
    result = 0
    l = 0
    for i in range(0, len(Actual)):
        if pred[i] != 0.0:
            result += abs((Actual[i])-(pred[i]))
            l+= 1

    result1 = result/ l
    return(result1)

def rmse_1(y_true, y_pred): # we only consider predicted ratings
    result = 0
    l = 0
    for i in range(0, len(y_true)):
        if y_pred[i] != 0.0:
            result += ((y_true[i])-(y_pred[i]))**2
            l += 1
    result1 = sqrt((result)/ l)
    return(result1)


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
        conf = confusion(actual, predicted, 3)
        cov = coverage(actual, predicted)
        prec = precision(conf[0], conf[1])
        rec = recall(conf[0], conf[3])
        f_msure = f_measure(prec, rec)
        result.append([mae_1(actual, predicted),rmse_1(actual, predicted), prec,rec, f_msure,cov] )

    scores = moy_metric(result)
    print(scores)
    sys.stdout = open("results\cosine\pred_new_Movielens100kresult.txt", "a+")
    print("resultas avec un paramètre k =  " + str(50) + " : ")
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
    precision = 0.0
    recall = 0.0
    cov =0.0
    f_mesure = 0.0
    for j in range(0,len(scores)):
        mae += scores[j][0]
        rmse += scores[j][1]
        precision += scores[j][2]
        recall += scores[j][3]
        f_mesure += scores[j][4]
        cov += scores[j][5]

    mae = mae / len(scores)
    rmse = rmse / len(scores)
    precision = precision /len(scores)
    recall = recall /len(scores)
    f_measure = f_mesure /len(scores)
    cov = cov / len(scores)
    return ([mae,rmse,precision,recall, f_measure,cov])
# ----------------- end  prepare test and train sets for 5-cross validation -------------------
def confusion(y_true, y_pred, threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    #print('len du valeurs {}'.format(len(y_true)))
    for i in range(0, len(y_true)):
        #print('y_pred= {} y_true={}'.format(y_pred[i], y_true[i]))

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
    #print("pt is {}".format(pt))
    return result
# ----------------- begin similarity and neighborhood selection CF   ---------------------

"""
check_neighbors_validation
this function returns (from a nearest neighbors list) users who rated the target item
"""

def check_neighbors_validation(train, movie_id):
    valid_neighbors = []
    rated = []
    #rated = train.loc[:, movie_id]
    rated = train.get(movie_id)
    #print(rated)
    if rated is not None:
        for i in range(0, len(rated)):
            if int(rated.values[i]) != 0.0:
                valid_neighbors.append(rated.index.values[i])
    else:
        valid_neighbors = []
    #print(len(rated))

    #print(valid_neighbors)
    return valid_neighbors

"""
cosine_sim
this function returns cosine similarity for 
a target user (in test_set) and all his neighbors (in train_set)
"""
def inter_rating(test,train, u1, u2):  # calculer l'intersection entre les votes de deux utilisateurs
    co_rated = []
    v1 = []
    v2 = []
    rated1 = rated_items(test, u1)
    rated2 = rated_items(train, u2)
    for i in range(0, len(rated1[0])):
        for j in range(0,len(rated2[0])):
            if rated1[0][i] == rated2[0][j]:
                co_rated.append(rated1[0][i])
                v1.append(rated1[1][i])
                v2.append(rated2[1][j])
    inter_rating =[co_rated,v1,v2]
    return(inter_rating)


def cosine_sim(test,train,u1,u2):  # retourne les similarités cosinus des utilisateurs par  rapport à user_id
    vec1 = str_list_int(test.loc[u1].tolist())
    vec2 = str_list_int(train.loc[u2].tolist())
    co_rated1 = inter_rating(test, train, u1, u2)[1]
    co_rated2 = inter_rating(test, train, u1, u2)[2]
    if co_rated1 ==  []: return 0
    cosine_similarity = dot(co_rated1, co_rated2) / (norm(vec1) * norm(vec2))
    return(cosine_similarity)

"""
str_list_int
this function converts a str list to int list
"""

def str_list_int(list):     # convertir une liste de STR à une list de INT
    for i in range(0, len(list)):
        list[i] = int(list[i])
    return(list)


def k_nearest_neighbors(test, train, user_id, item_id, k, distance):
    similarity = []
    neighbors = check_neighbors_validation(train, item_id)
    for i in range(0, len(neighbors)):
        user = neighbors[i]
        sim = distance(test, train, user_id, user)
        if sim != 0:
            similarity.append([user, sim])
    similarity = sorted(similarity, key=lambda x: (-x[1]))
    #print(similarity[:k])
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
def predict_rating_new(test, train, user_id, item_id, l,distance):
    top_res = 0
    but_res = 0
    # print("-------------------  k_valid_nearest_neighbor  ---------------------------"
    nearest_neighbors = k_nearest_neighbors(test, train, user_id, item_id, l, distance)

    if not len(nearest_neighbors):
        return 0.0

    r_true = int(test.loc[user_id][item_id])  # added line
    test.loc[user_id][item_id] = 0  # added line
    r_target_moy = ratings_moy(test, user_id)  # added line
    #r_target_moy = ratings_moy(test, user_id)
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

    #print(pred)
    test.loc[user_id][item_id] = float(r_true)
    return pred
# ----------------- end prediction   ---------------------

"""
evaluate_algorithm_dataframe
This function load evaluation data for each fold, perform prediction using algorithm
and save results MAE and RMSE in txt file
"""

def evaluate_algorithm_dataframe(algorithm, distance,dataset_name, fold,*args):
    start_time = time.time()
    print("i am in evaluate")
    predicted = []
    test_set = load_eval_data("test_set", fold)
    train_set = load_eval_data("train", fold)
    pairs = load_eval_data("pairs", fold)
    actual = load_eval_data("actual", fold)

    kll=0
    print("Number of pairs to predit = {}".format(len(pairs)))

    for i in range (0,len(pairs)):
        begin = time.time()
        user_id = pairs[i][0]
        item_id = pairs[i][1]
        pre=algorithm(test_set,train_set,user_id, item_id, *args,distance)
        predicted.append(pre)
        print("user={} item={} kll={} actual={} predicted={} time={}".format(user_id, item_id, kll, actual[kll], pre,time.time() - begin))
        kll=kll+1
    #print(len(predicted))
    #print(predicted)

    path = "testtrain\Movielens100k\\fold"+str(fold)
    outfile = open(path +"\\predicted_new"+ str(fold), 'wb')
    pickle.dump(predicted, outfile)
    outfile.close()

    scores = [mae_1(actual, predicted),rmse_1(actual, predicted)]
    sys.stdout = open("testtrain\Movielens100k\\new_result_cosine_fold"+str(fold)+".txt", "a+")
    print("resultas pour  "+str(distance)+"  avec un paramètre k =  "+str(*args)+" : ")
    print("accuracy")
    print(scores)
    print("That took {} seconds".format(time.time() - start_time))
    sys.stdout.close()

    return(scores)


# -----------------*******************    main    *********************--------------------------------
if __name__ == '__main__':
    sparsity_thershold=0.985
    #compute_evaluate()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        train_set = load_eval_data("train", 4)
        sparsity_level = sparsity(train_set)
        executor.submit(evaluate_algorithm_dataframe(predict_rating_new, improved_sim,"Movielens100k",4,50))
