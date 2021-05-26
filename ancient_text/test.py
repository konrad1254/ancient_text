from pre_processing import Preprocessor
from genetic_programming import Genetic

import pandas as pd
import matplotlib.pyplot as plt
import re


text = "Undique loci natura Helvetii continentur: Una ex parte flumine Rheno latissimo atque altissimo, qui agrum Helvetium a Germanis dividit; altera ex parte monte Iura altissimo, qui est inter Sequanos et Helvetios; tertia lacu Lemanno et flumine Rhodano, qui provinciam nostram ab Helvetiis dividit. \
    His rebus fiebat ut et minus late vagarentur et minus facile finitimis bellum inferre possent.\
Qua ex parte homines bellandi cupidi magno dolore adficiebantur.\
Pro multitudine autem hominum et pro gloria belli atque fortitudinis angustos se fines habere arbitrabantur, \
qui in longitudinem milia passuum CCXL, in latitudinem CLXXX patebant."

data = text.split()

pre = Preprocessor(tokens = data, remove_list = ['gloria', 'mont'], lemmatize = True)
print(pre.cleaning())

data = pre.cleaning()

genetic_algorithm = Genetic(data = data, numberOfParents = 5, generations = 5, no_of_cr = 2, childSize = 4, 
							prob_of_mutation = 0.1, lambda_fitness = 100, num_topics_bounds = (2,7), alpha_choice=['symmetric', None, 'asymmetric'], eta_optimize = False)

model, return_dict, model_corpus, num_topics, tracker_output = genetic_algorithm.fit()

topic_corpus = model[model_corpus]
# get the topic descritions
topic_sep = re.compile(r"0\.[0-9]{3}\*") # getting rid of useless formatting
# extract a list of tuples with topic number and descriptors from the model
model_topics = [(topic_no, re.sub(topic_sep, '', model_topic).split(' + ')) for topic_no, model_topic in
                model.print_topics(num_topics=num_topics, num_words=10)]

descriptors = []
topic_numbers = []
for i, m in model_topics:
    print(i+1, ", ".join(m[:10]))
    topic_numbers.append('Topic #' +str(i+1))
    descriptors.append(", ".join(m[:2]).replace('"', ''))

target_category = data.keys()
# get a list of all the topic scores for each document
scores = [[t[1] for t in topic_corpus[entry]] for entry in range(len(model_corpus))]
# turn that into a data frame with N rows and K columns, each with the score of the corresponding topic
topic_distros = pd.DataFrame(data=scores, columns=descriptors)
# add the review category of each document (so we can aggregate)
topic_distros['category'] = target_category

plt.figure()
fig, ax = plt.subplots(figsize=(10, 5)) # set graph size
# aggregate topics by categories
aggregate_by_category = topic_distros.groupby('category').mean()
# plot the graph
aggregate_by_category[descriptors].plot.bar(ax=ax);
# move the legend out
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))

plt.legend(by_label.values(), topic_numbers, loc='top right')
plt.show()