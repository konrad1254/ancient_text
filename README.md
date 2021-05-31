# Ancient Text
Repository for NLP analysis of ancient text

## Installation & Setup 
```bash
pip3 install git+https://github.com/konrad1254/ancient_text@main
```
## Reading Data
- reading data from pdf
- collecting a Latin-only text

## Cleaning Data
The package supports text cleaning. Taking a list of tokens, the cleaning objects removes digits, punctuation, stop words as well as pre-specified words.
Additionally, it supports lemmatization.

Given a list of tokens, the cleaning is conducted in the following way:

```python
from ancient_text import pre_processing
pre = pre_processing.Preprocessor(tokens = data, remove_list = [], lemmatize = True)
data = pre.cleaning()
```
Arguments:
- tokens
- remove_list: additional words to be removed besides Latin stop words
- lemmatize: should the token be lemmatized

## Word Clouds
The word cloud generator is simply recalled by the following:

```python
from ancient_text import vizualisations
vizualisations.word_cloud_generator(data, 'test')
```
Arguments:
- data (tokens)
- name under which .png is saved. The location corresponds to cwd.

## LDA with Genetic Programming
Topic modelling can be conducted in the following way:
```python
from ancient_text import genetic_programming
genetic_algorithm = genetic_programming.Genetic(data = data, 
                                                numberOfParents = 5, 
                                                generations = 5, 
                                                no_of_cr = 2, 
                                                childSize = 4, 
                                                prob_of_mutation = 0.1, 
                                                lambda_fitness = 100, 
                                                num_topics_bounds = (2,7), 
                                                alpha_choice=['symmetric', None, 'asymmetric'], 
                                                eta_optimize = False)

model, return_dict, model_corpus, num_topics, tracker_output = genetic_algorithm.fit()
```
Arguments:
- data: either list of tokens or dictionary with values of list of tokens
- numberOfParents / generations / no_of_cr / childSize / prob_of_mutation: genetic programming parameters
- lambda_fitness: weight of stability factor
- num_topics_bounds: (min,max) of number of topic bounds
- alpha_choice: choices of alpha hyperparameter
- eta_optimize: choice if eta should be optimized or not 

Output: most importantly model artifact



