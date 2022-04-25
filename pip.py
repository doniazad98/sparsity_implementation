import numpy as np
import pandas as pd
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

def corated_items(test,train, user_id, v):
    co_rated = []
    rated_by_u = rated_items(test,user_id)
    rated_by_v = rated_items(train,v)
    for i in range(0, len(rated_by_u[0])):
        for j in range(0, len(rated_by_v[0])):
            if rated_by_u[0][i] == rated_by_v[0][j]:
                co_rated.append(rated_by_u[0][i])
    return co_rated

def moyen_rating_item(test,train,i):
    sum=0
    num=0
    u_test=test.index.values
    v_train=train.index.values
    for u in u_test :
        if test.loc[u,i]!=0:
            sum=sum+test.loc[u,i]
            num=num+1
    for v in v_train:
        if train.loc[v,i]!=0:
            sum = sum + train.loc[v, i]
            num = num + 1
    #print("moyen ratings item={}".format(sum/num))
    if num==0:
        return 0
    else :
        return sum/num

def sim_pip(test,train,u,v):
    co_items=corated_items(test,train,u,v)
    sim=0
    for i in co_items:
        uk = moyen_rating_item(test, train,i)
        #print("u={} v={} i={}".format(u,v,i))
        sim=sim+pip(test,train,test.loc[u,i],train.loc[v,i],uk)
    print("similarity final = {}".format(sim))
    return sim

def pip(test,train,ru,rv,uk):
    im=impact(ru,rv)
    pro=proximity(ru,rv)
    pop=popularity(ru,rv,uk)
    #print("ru={} rv={}".format(ru,rv))
    print("Impact={} proximity={} popylarity={}".format(im ,pro,pop))

    return im*pro*pop

def Agreement(ru,rv):
    if (ru > rmed > rv) or (ru < rmed < rv) :
        return False
    else :
        return True

def impact(ru,rv):
    im=(abs(ru-rmed) +1)*(abs(rv-rmed)+1)
    if Agreement(ru,rv):
        return im
    else :
        if im != 0:
            return 1 / im
        else:
            return 0

def proximity(ru,rv)  :
    if Agreement(ru,rv):
        distance =abs(ru-rv)
    else:
        distance=2*abs(ru-rv)
    val=2*(r_max-r_min)+1
    return (val-distance)*(val-distance)

def popularity(ru,rv,uk):
    if (ru>uk and rv>uk) or (ru<uk and rv<uk):
        return 1+ (((ru+rv)/2)-uk)*(((ru+rv)/2)-uk)
    else:
        return 1

def piparticle(test,train,user_id):
    users = train.index.values
    similarity=[]
    for v in users :
        pip=sim_pip(test,train,user_id,v)
        similarity.append([v,pip])
    #print(" -------------------------------- getting out  sparcity aware")
    print("similarity= {}".format(similarity))
    return similarity

if __name__ == '__main__':
    r_max=5
    r_min=1
    rmed=(r_max+r_min)/2
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
    #moyen_rating_item(data_fr,data_fr,'item1')
    users=data_fr.index.values
    piparticle(data_fr,data_fr,'User1')