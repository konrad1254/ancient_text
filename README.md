# Ancient Text
Repository for NLP analysis of ancient text

## Installation & Setup 
Under construction

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

