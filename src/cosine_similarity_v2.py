from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

most_visited = {"John": "London Paris London Denver London Denver London Paris London Denver London Denver",
        "Sean": "Paris Denver Paris Paris London London", 
        "Mary": "Paris Denver Paris Paris Paris Paris", 
        "Shawna": "Toronto Calgary Toronto", 
        "Jonas": "Vancouver Toronto Toronto London"}

print("\n")
print("--------------------------------------------------")
print("{0:10s}: {1:s}".format("Traveller","Cities visited"))
print("--------------------------------------------------")
for traveller, cities in most_visited.items():
    print("{0:10s}: {1:s}".format(traveller,cities))
print("--------------------------------------------------\n")

cv = CountVectorizer()
cv_fit = cv.fit_transform(most_visited.values())
print(pd.DataFrame(data=cv_fit.toarray(), index=most_visited.keys(), columns=cv.get_feature_names()))
print("\n")
#print(cv_fit.toarray())

print ('Shape of Sparse Matrix: ', cv_fit.shape)
print ('Amount of Non-Zero occurences: ', cv_fit.nnz)
print ('sparsity: %.2f%%' % (100.0 * cv_fit.nnz /
                             (cv_fit.shape[0] * cv_fit.shape[1])))
print("\n")
#print(cv.get_feature_names())
cs = cosine_similarity(cv_fit.toarray())
sim = pd.DataFrame(cs, index=most_visited.keys(), columns=most_visited.keys())
#sim.columns = cv.get_feature_names()
#sim.index = cv.get_feature_names()
print(sim)

print("\n")
c1 = (np.asarray(cv_fit.sum(axis=0))[0])
#print(c1.tolist())
c2 = list(cv.get_feature_names())
#print(c2)
frq = pd.DataFrame(dict(City=c2, Visits=c1))
print(frq)
