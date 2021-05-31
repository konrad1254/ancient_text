# Ancient Text
This is a repository supporting NLP research in ancient texts. 
The repo supports reading, cleaning, vizualisation and modelling of textual data.

The topic model is based on a combination of genetic programming and Latent Dirichlet allocation.

## Installation & Setup 
```bash
pip3 install git+https://github.com/konrad1254/ancient_text@main
```
Note that there are quite a few external dependencies; this might take a while.

## Reading Data
Reading complicated, non-machine readable text data is made easy.

```python
import re
from ancient_text import utils
from ancient_text import reading_data

path = '/Users/konrad/Documents/test'
document = '/Users/konrad/Documents/test/roll_extract.pdf'

data = reading_data.magic_converter(document, path)
files = utils.find_txt_filenames(path)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

sorted_files = sorted(files, key = natural_keys)
raw_text = utils.string_conversion(sorted_files)
utils.cleaning_pdf_output(raw_text)
```
The package extracts the PDF file into raw txt files per page. Then, the files are ordered and loaded into the python environment. 
Lastly, the files are converted and rudimentarily cleaned.

## Language Extraction
So far, the package supports topic modelling for Latin. 

```python
from ancient_text import utils
language_extraction(text)
```

## Cleaning Data
The package supports text cleaning. Taking a list of tokens, the cleaning objects removes digits, punctuation, stop words as well as pre-specified words.
Additionally, it supports lemmatization.

Given a list of tokens, the cleaning is conducted in the following way:

```python
from ancient_text import pre_processing
from ancient_text import utils
utils.get_corpora()

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



