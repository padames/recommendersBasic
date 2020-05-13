import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import math
from math import isnan


plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=True)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('font', size=12) 



# Load Data set
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../data/ml-100k/u.user', sep='|', names=u_cols, encoding='iso-8859-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=r_cols,  encoding='iso-8859-1')

# the movies file contains columns indicating the movie's genres
# let's only load the first three columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date']
movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=m_cols, usecols=range(3), encoding='iso-8859-1')

# Construccion del DataFrame
data = pd.merge(pd.merge(ratings, users), movies)
data = data[['user_id','title', 'movie_id','rating','release_date','sex','age']]


print("The movielens database has\n"
    +"    " + str(data.shape[0]) + " ratings\n"
    +"      ", data.user_id.nunique(),"users\n"
    +"     ", data.movie_id.nunique(), "movies.")

print(data.head())

# dataframe with the data from user 1
data_user_1 = data[data.user_id==1]
# dataframe with the data from user 2
data_user_2 = data[data.user_id==6]
# We first compute the set of common movies
common_movies = set(data_user_1.movie_id).intersection(data_user_2.movie_id)
print( "\nNumber of common movies",len(common_movies),'\n')

# creat the subdataframe with only with the common movies
mask = (data_user_1.movie_id.isin(common_movies))
data_user_1 = data_user_1[mask]
print(data_user_1[['title','rating']].head())

mask = (data_user_2.movie_id.isin(common_movies))
data_user_2 = data_user_2[mask]
print(data_user_2[['title','rating']].head())

from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
import seaborn as sns;

def SimEuclid(dataFrame, user1, user2, min_common_items=1):
    """Returns a distance-based similarity score for person1 and person2
    """
    movies_user1 = dataFrame[dataFrame['user_id'] == user1 ]
    movies_user2 = dataFrame[dataFrame['user_id'] == user2 ]
    
    rep = pd.merge(movies_user1, movies_user2, on='movie_id')    
    
    if len(rep) == 0 or (len(rep) < min_common_items):
        return 0
    else:
        return 1.0 / (1.0 + euclidean(rep['rating_x'], rep['rating_y'])) 


def SimPearson(dataFrame, user1, user2, min_common_items=1):
    """Returns a pearsonCorrealation-based similarity score for 
    user1 and user2
    """
    # GET MOVIES OF USER1
    movies_user1=dataFrame[dataFrame['user_id'] == user1 ]
    # GET MOVIES OF USER2
    movies_user2=dataFrame[dataFrame['user_id'] == user2 ]
    
    # FIND SHARED FILMS
    rep=pd.merge(movies_user1, movies_user2, on='movie_id')
    if len(rep)==0:
        return 0    
    if(len(rep) < min_common_items):
        return 0
    if len(rep['rating_x']) < 2 or len(rep['rating_y']) < 2:
        return 0
    res = pearsonr(rep['rating_x'], rep['rating_y'])[0]
    if(isnan(res)):
        return 0
    else:
        return res
    
    
print("""
|Distance Measure|sim(1,8)|sim(1,31)|
|Euclidean|{0}|{1}|
|Pearson|{2}|{3}|"""
    .format(round(SimEuclid(data, 1, 8), 4), round(SimEuclid(data, 1, 31), 4),\
            round(SimPearson(data, 1 ,8), 4), round(SimPearson(data, 1, 31), 4)))

movies_user1=data[data['user_id'] ==1 ][['user_id','movie_id','rating']]
movies_user2=data[data['user_id'] ==8 ][['user_id','movie_id','rating']]
    
# FIND SHARED FILMS
rep=pd.merge(movies_user1 ,movies_user2,on='movie_id')
x = rep.rating_x + np.random.normal(loc=0.0, scale=0.1, size=len(rep.rating_x))
y = rep.rating_y + np.random.normal(loc=0.0, scale=0.1, size=len(rep.rating_y))
    
a = rep.groupby(['rating_x', 'rating_y']).size()
x = []
y = []
s = []
for item,b in a.iteritems():
    x.append(item[0])
    y.append(item[1])
    s.append(b*30)

fig = plt.figure(figsize=(6, 4))
plt.scatter(x, y, s=s)
plt.xlabel('Rating User 1')
plt.ylabel('Rating User ' + str(8))
plt.axis([0.5,5.5,0.5,5.5])
plt.savefig("../images/corre18.png", dpi= 300, bbox_inches='tight')
plt.show()


movies_user1=data[data['user_id'] ==1 ][['user_id','movie_id','rating']]
movies_user2=data[data['user_id'] ==31 ][['user_id','movie_id','rating']]
    
# FIND SHARED FILMS
rep = pd.merge(movies_user1, movies_user2, on='movie_id')
x = rep.rating_x + np.random.normal(loc=0.0, scale=0.1, size=len(rep.rating_x))
y = rep.rating_y + np.random.normal(loc=0.0, scale=0.1, size=len(rep.rating_y))
    
a = rep.groupby(['rating_x', 'rating_y']).size()
x = []
y = []
s = []

fig = plt.figure(figsize=(6,4))
for item,b in a.iteritems():
    x.append(item[0])
    y.append(item[1])
    s.append(b*30)
plt.scatter(x, y, s=s)
plt.xlabel('Rating User 1')
plt.ylabel('Rating User ' + str(31))
plt.axis([0.5,5.5,0.5,5.5])
plt.savefig("../images/corre131.png", dpi=300, bbox_inches='tight')
plt.show()


def assign_to_set(df):
    sampled_ids = np.random.choice(df.index,
                                   size=np.int64(np.ceil(df.index.size * 0.2)),
                                   replace=False)
    df.loc[sampled_ids, 'for_testing'] = True
    return df

data['for_testing'] = False
groupped = data.groupby('user_id', group_keys=False).apply(assign_to_set)
# use mask to separate train and test sets
data_train = data[groupped.for_testing == False]
data_test = data[groupped.for_testing == True]
print (data_train.shape)
print (data_test.shape)
print (data_train.index & data_test.index)

print ("The training data set has " + str(data_train.shape[0]) + " ratings")
print ("The test data set has " + str(data_test.shape[0]) + " ratings")
print ("The data set has " + str(data.movie_id.nunique()) + " movies")


def compute_rmse(y_pred, y_true):
    """ Compute Root Mean Squared Error. """
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))


class CollaborativeFiltering:
    """ Collaborative filtering using a custom sim(u,u'). """
    
    def __init__(self, DataFrame, similarity=SimPearson):
        """ Constructor """
        self.sim_method = similarity # Gets recommendations for a person by using a weighted average
        self.df = DataFrame
        self.sim = pd.DataFrame(np.sum([0]),columns=data_train.user_id.unique(), index=data_train.user_id.unique())

    def learn(self):
        """ Prepare data structures for estimation. Similarity matrix for users """
        allUsers=set(self.df['user_id'])
        self.sim = {}
        for person1 in allUsers:
            self.sim.setdefault(person1, {})
            a = data_train[data_train['user_id']==person1][['movie_id']]
            data_reduced = pd.merge(data_train, a, on='movie_id')
            for person2 in allUsers:
                # no es comparem am nosalres mateixos
                if person1==person2: 
                    continue
                self.sim.setdefault(person2, {})
                if(person1 in self.sim[person2]):
                    continue # since is a simetric matrix
                sim = self.sim_method(data_reduced, person1, person2)
                if(sim<0):
                    self.sim[person1][person2]=0
                    self.sim[person2][person1]=0
                else:
                    self.sim[person1][person2]=sim
                    self.sim[person2][person1]=sim
                
    def estimate(self, user_id, movie_id):
        totals={}
        movie_users=self.df[self.df['movie_id'] ==movie_id]
        rating_num=0.0
        rating_den=0.0
        allUsers=set(movie_users['user_id'])
        for other in allUsers:
            if user_id==other: 
                continue 
            rating_num += self.sim[user_id][other] * float(movie_users[movie_users['user_id']==other]['rating'])
            rating_den += self.sim[user_id][other]
        if rating_den==0: 
            if self.df.rating[self.df['movie_id']==movie_id].mean()>0:
                # return the mean movie rating if there is no similar for the computation
                return self.df.rating[self.df['movie_id']==movie_id].mean()
            else:
                # else return mean user rating 
                return self.df.rating[self.df['user_id']==user_id].mean()
        return rating_num/rating_den
    
import datetime.datetime.now as now

time1 =now()
reco = CollaborativeFiltering(data_train)
time2 = now()
reco.learn()
time3= now()
reco.estimate(user_id=2, movie_id=1)    
time4 = now()