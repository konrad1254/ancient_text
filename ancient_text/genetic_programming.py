import numpy as np
import pandas as pd
import re
import random 
from gensim.models import LdaMulticore, TfidfModel, CoherenceModel
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.metrics import jaccard_score
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore',module='gensim')  

class Genetic:
    """
    Class that produces genetic programming optimization for LDA
    """

    def __init__(self, data,numberOfParents, generations, no_of_cr, childSize, prob_of_mutation, lambda_fitness, num_topics_bounds, alpha_choice, eta_optimize = True):
        self.data = data
        self.numberOfParents = numberOfParents
        self.model = None
        self.population = dict()
        self.model_corpus = []
        
        # Hyperparameters to tune
        self.num_topics = np.empty([self.numberOfParents, 1])
        self.num_topic_min = num_topics_bounds[0]
        self.num_topic_max = num_topics_bounds[1]

        self.eta_optimize  = eta_optimize
        self.eta = np.array([0])
        
        self.alpha_choice  = alpha_choice
        self.alpha = list()
        
        self.decay = np.empty([self.numberOfParents, 1])
        self.offset = np.empty([self.numberOfParents, 1])

        # Hierarchical modelling: log-normal hyper-priors
        if self.eta_optimize == True:
            self.sigma_eta = np.empty([self.numberOfParents, 1])
            self.mu_eta = np.empty([self.numberOfParents, 1])
        else:
            self.eta = None

        self.generations = generations
        self.no_of_cr = no_of_cr
        self.childSize = childSize 
        self.prob_of_mutation = prob_of_mutation
        self.lambda_val = lambda_fitness



    def initalize(self, num_of_words):
        """
        
        """
        population = dict()
        if self.eta_optimize == True:
            self.eta = np.empty([num_of_words, self.numberOfParents])
        
        for i in range(self.numberOfParents):

            self.num_topics = int(round(random.uniform(self.num_topic_min, self.num_topic_max)))
            self.decay[i] = float(random.uniform(0.5, 1))
            self.offset[i] = float(random.uniform(0, 2))
            self.alpha.append(np.random.choice(self.alpha_choice, 1)) # allow for model choice 

            if self.eta_optimize == True:
                self.mu_eta[i] = float(random.uniform(0, 5))
                self.sigma_eta[i] = float(random.uniform(0, 5))
                parameter_eta = np.random.lognormal(self.mu_eta[i], self.sigma_eta[i], num_of_words).T
                self.eta[:,i] = np.random.dirichlet((parameter_eta), 1)
            
        population['eta'] = self.eta
        population['num_topics'] = self.num_topics
        population['alpha'] = self.alpha
        population['decay'] = self.decay
        population['offset'] = self.offset

        return population

    def data_prep(self):
        """
        
        """
        return_dict = {}
        text = []
        out_of_sample = []
        in_sample = []
        iterator =  0


        if isinstance(self.data, dict):

            out_of_sample_keys = np.random.choice(list(self.data.keys()), int(round(0.2*len(self.data.keys()))))

            for k in self.data.keys():
                text.append(self.data[k]) 
                if k in out_of_sample_keys:
                    out_of_sample.append(self.data[k])
                else:
                    in_sample.append(self.data[k])

        elif isinstance(self.data, list):
            cut_off = int(round(0.8*len(self.data)))
            out_of_sample = text[cut_off:]
            in_sample = text[:cut_off]
            text = self.data

        else:
            raise NotImplementedError

        dictionary = Dictionary(text)
        dictionary.filter_extremes(no_below=3, no_above=0.7)
        ldacorpus = [dictionary.doc2bow(i) for i in text]
        tfidfmodel = TfidfModel(ldacorpus)
        model_corpus = tfidfmodel[ldacorpus]

        # In-sample
        dictionary_in_s = Dictionary(in_sample)
        dictionary_in_s.filter_extremes(no_below=3, no_above=0.7)
        ldacorpus_is = [dictionary_in_s.doc2bow(i) for i in text]
        tfidfmodel_is = TfidfModel(ldacorpus_is)
        model_corpus_is = tfidfmodel_is[ldacorpus_is]

        return_dict['model_corpus'] = model_corpus
        return_dict['model_corpus_is'] = model_corpus_is
        return_dict['dictionary_is'] = dictionary_in_s
        return_dict['dictionary'] = dictionary
        return_dict['in_sample'] = in_sample
        return_dict['out_of_sample'] = out_of_sample

        return return_dict

    def stability_score(self, num_topics, eta, alpha, decay, offset):

        n = 10 # number of stability tests
        i = 0
        store_topic_list_of_list = []
        for i in range(n):
            random_state = i
            model, topic_corpus = self.lda_stability_test(num_topics, eta, alpha, decay, offset, random_state = random_state)    

            # get the topic descritions
            topic_sep = re.compile(r"0\.[0-9]{3}\*") # getting rid of useless formatting
            # extract a list of tuples with topic number and descriptors from the model
            model_topics = [(topic_no, re.sub(topic_sep, '', model_topic).split(' + ')) for topic_no, model_topic in
                            model.print_topics(num_topics=num_topics, num_words=10)]

            descriptors = []
            for i, m in model_topics:
                descriptors.append(m[:10])
            store_topic_list_of_list.append(descriptors)

        # store_topic_list_of_list[run][topic]
        cleaned_list = []
        for i in range(len(store_topic_list_of_list)):
            list_per_rurn = [item for sublist in store_topic_list_of_list[i] for item in sublist]
            cleaned_list.append([i.replace('"', '') for i in list_per_rurn])

        running = []
        for i in range(len(cleaned_list)):
            word_set = set(cleaned_list[i])
            for j in range(len(cleaned_list)):
                if i !=j:
                    word_set1 = cleaned_list[i]
                    word_set2 = cleaned_list[j]

                    score = jaccard_score(list(word_set1), list(word_set2), average='micro')

                    running.append(score)

        return np.mean(running)

    def fitenss(self, dictionary, out_of_sample, model, num_topics, eta, alpha, decay, offset):
        """
        Evaluate the fitness score of the model (UMASS)
        """
        lambda_parameter = self.lambda_val
        coherencemodel_umass = CoherenceModel(model=model, 
                                          texts=out_of_sample, 
                                          dictionary=dictionary, 
                                          coherence='u_mass')

        umass_score = coherencemodel_umass.get_coherence()
        stability = self.stability_score(num_topics, eta, alpha, decay, offset)

        return umass_score + lambda_parameter*stability
    
    def genetic_stochastic_movement(self, num_of_words):
        """
        Adding stochastic movement between generations
        """
        population = dict()

        for i in range(self.numberOfParents):
            stochastic_val_topic = int(round(random.uniform(-1, 1)))
            if self.num_topics + stochastic_val_topic > self.num_topic_min and self.num_topics + stochastic_val_topic < self.num_topic_max:
                self.num_topics = self.num_topics + stochastic_val_topic

            if self.eta_optimize == True:
                self.mu_eta[i] = self.mu_eta[i] + np.random.normal(0,0.1,1)
                stochastic_pert_sigma = np.random.normal(0,0.1,1)
                if self.sigma_eta[i] + stochastic_pert_sigma > 0:
                    self.sigma_eta[i] = self.sigma_eta[i] + stochastic_pert_sigma
            
                parameter_eta = np.random.lognormal(self.mu_eta[i], self.sigma_eta[i], num_of_words).T
                self.eta[:,i] = np.random.dirichlet((parameter_eta), 1)

            stochastic_pert_decay = np.random.normal(0,0.1,1)
            if (self.decay[i] + stochastic_pert_decay > 0) and (self.decay[i] + stochastic_pert_decay < 0.99):
                self.decay[i] = self.decay[i] + stochastic_pert_decay

            stochastic_pert_offset = np.random.normal(0,0.05,1)
            if (self.offset[i] + stochastic_pert_offset > 0):
                self.offset[i] = self.offset[i] + stochastic_pert_offset            

        population['eta'] = self.eta
        population['num_topics'] = self.num_topics
        population['alpha'] = self.alpha
        population['decay'] = self.decay
        population['offset'] = self.offset

        return population
    

    def train_population(self, generation):
        """
        Training initial population
        """

        prepared_data = self.data_prep()

        if generation == 0:
            population = self.initalize(len(prepared_data['dictionary']))
        else:
            population = self.genetic_stochastic_movement(len(prepared_data['dictionary']))

        score = []
        result_dict = dict()
        identifier = ''
        counter_ident = 0
        for i in range(self.numberOfParents):
            identifier = str(counter_ident)
            if self.eta_optimize == True:  
                eta = self.eta[:,i]
            else:
                eta = None

            alpha = self.alpha[i][0]
            num_topics  = self.num_topics
            decay = self.decay[i]
            offset = self.offset[i]

            result = (eta,alpha,num_topics,decay,offset)
            
            model = LdaModel(corpus = prepared_data['model_corpus'], id2word = prepared_data['dictionary'],
                            num_topics = num_topics, eta = eta, alpha = alpha, decay = decay, offset = offset,
                            iterations = 1000, random_state = 42)

            result_dict[identifier] = result

            fitness = self.fitenss(prepared_data['dictionary'], prepared_data['out_of_sample'], model, num_topics, eta, alpha, decay, offset)
            score.append(fitness)
            counter_ident += 1

        return result_dict, prepared_data['model_corpus'],  prepared_data['dictionary'], prepared_data['out_of_sample'], score

    def crossover_uniform(self, result_dict, no_of_cr, childSize, score):
        '''
        Mate parents to create children having parameters from these parents (we are using uniform crossover method)
        '''
        max_list = sorted(range(len(score)), key=lambda k: score[k])
            
        cr = []
        for j in range(childSize): 
            i = max_list[j]
            cr.append(result_dict[str(i)])  
            
        
        children_dict = dict() 
        for i in range(childSize):
            eta = cr[np.random.choice(2)][0]
            alpha = cr[np.random.choice(2)][1]
            num_topics = cr[np.random.choice(2)][2]
            decay = cr[np.random.choice(2)][3]
            offset = cr[np.random.choice(2)][4]

            children_dict[i] = (eta,alpha,num_topics,decay,offset)
        
        return children_dict

    def mutation(self,children_dict, prob_of_mutation, dictionary):
        """
        Introduce random mutations
        """

        return_dict = dict()

        # number of topics
        num_topics = self.num_topics

        # Possible mutations in alpha, eta and decay
        for child, values in children_dict.items():
            
            val = list(values)
            # eta
            if self.eta_optimize == True:
                if  prob_of_mutation > float(np.random.uniform(0,1,1)):
                    mu_eta = int(round(random.uniform(0.1, 1)))
                    sigma_eta = float(random.uniform(0.1, 2))
                    parameter_eta = np.random.lognormal(mu_eta, sigma_eta, int(len(dictionary))).T
                    val[0] = np.random.dirichlet((parameter_eta), 1).reshape((len(dictionary),))
            
            # alpha
            #if  prob_of_mutation > float(np.random.uniform(0,1,1)):
            #    val[1] = np.random.choice(self.alpha_choice, 1)
            
            # decay
            if  prob_of_mutation > float(np.random.uniform(0,1,1)):
                val[3] = float(random.uniform(0.5, 1))

            # offset
            if  prob_of_mutation > float(np.random.uniform(0,1,1)):
                val[4] = float(random.uniform(0, 2))


            return_dict[child] = val
        
        return return_dict

    
    def genetic_programming(self, generation):
        """
        Run LDA over children and return the best
        """
        result, corpus_training, dictionary, out_of_sample, score = self.train_population(generation)
        children_dict = self.crossover_uniform(result, self.no_of_cr, self.childSize, score)
        children = self.mutation(children_dict, self.prob_of_mutation, dictionary)

        output = dict()

        for key, value in children.items():
            eta = value[0]
            alpha = value[1]
            num_topics = value[2]
            decay = value[3]
            offset = value[4]

            result = (eta,alpha,num_topics, decay, offset)

            model = LdaModel(corpus = corpus_training, id2word = dictionary,
                            num_topics = num_topics, eta = eta, alpha = alpha, decay = decay, offset = offset,
                            iterations = 1000, random_state = 42)

            output[self.fitenss(dictionary, out_of_sample, model, num_topics, eta, alpha, decay, offset)] = result

        return output, dictionary

    def fit(self):
        """
        Run LDA over all generations and returrn the best
        """
        return_values = dict()
        tracker_output = dict()
        for i in range(self.generations):
            print(f"Fitting generation: {i}")
            output, dictionary = self.genetic_programming(i)
            
            return_values[max(output.items(), key=lambda x: x[0])[0]] = max(output.items(), key=lambda x: x[0])[1]
            tracker_output[i] = output

        # final model
        metrics = max(return_values.items(), key=lambda x: x[0])[1]

        print(f'Best model found with score of {max(return_values.items(), key=lambda x: x[0])[0]}' )
        
        eta = metrics[0]
        alpha = metrics[1]
        num_topics = metrics[2]
        decay = metrics[3]
        offset = metrics[4]

        print(f'Parameters: #topics: {num_topics}')
        print(f'Parameters: decay: {decay}')
        print(f'Parameters: offset: {offset}')
        print(f'Parameters: alpha: {alpha}')

        model = LdaModel(corpus = self.model_corpus, id2word = dictionary,
                        num_topics = num_topics, alpha = alpha, eta = eta, decay = decay, offset = offset,
                         iterations = 1000, random_state = 42)

        return model, return_values, self.model_corpus, num_topics, tracker_output

    def data_return(self):
        prepared_data = self.data_prep()
        return prepared_data

    def lda_stability_test(self, num_topics, eta, alpha, decay, offset, random_state): 
        prepared_data = self.data_prep()

        model = LdaModel(corpus = prepared_data['model_corpus'], id2word = prepared_data['dictionary'], 
                        num_topics = num_topics, alpha = alpha, eta = eta, decay = decay, offset = offset,
                        iterations=1000, random_state = random_state) 
        
        topic_corpus = model[prepared_data['model_corpus']]

        return model, topic_corpus





    
