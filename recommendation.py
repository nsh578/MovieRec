import numpy
import scipy
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
#lightfm considers 5 as positive and 4< as negative to solve in binary.

#TWO TYPES OF RECOMMENDATION ALGS:
#1. Content-based: see what you liked in the past
#2. Collaborative: see what similar other users liked in the past
#3. Hybrid (lightfm)


#fetch data and format it
#fetch_movielens is csv file with 100k+ movie ratings on 1700 movies
#create interaction matrix from csv file and store it as dictionary to var data
#Matrix: row: user, column: movie, value: rating
data = fetch_movielens(min_rating=4.0)

#print
print(repr(data['train'])) #shows # of training data (about x10 more than test)
print(repr(data['test'])) #shows # of testing data 

#CREATE MODEL
#loss measures difference between model's prediction & desired output
#warp: Weighted Approximate-Rank Pairwise
#-create recommendation by existing user rating pairs
model = LightFM(loss='warp')

#TRAIN MODEL
#set #of epochs and parellel computing threads
model.fit(data['train'], epochs=30, num_threads=2)

print(data)
print(data['item_labels'])

def sample_recommendation(model, data, user_ids):

	#number of users and items(movies) in training data using shape attr
	n_users, n_items = data['train'].shape

	#generate recs for each user we input
	for user_id in user_ids:

		#movies they already like
		#compressed sparse row format
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#movies the model predicts they will like
		#arange method gives all nums 0 - n_items and predict score for every movie
		scores = model.predict(user_id, np.arange(n_items))
		#rank them in order of most liked to least
		#argsort with negative sign sorts in descending order
		top_items = data['item_labels'][np.argsort(-scores)]

		#print out results
		print("User %s" %user_id)
		print("     Known positives:")

		for i in known_positives[:3]:
			print("           %s" % i)

		print("    Recommended:")

		for i in top_items[:3]:
			print("           %s" % i)

sample_recommendation(model, data, [1, 3, 25, 107, 450])