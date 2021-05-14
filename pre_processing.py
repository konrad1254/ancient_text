### Implement NLP cleaning here
from cltk.tokenize.word import WordTokenizer
from cltk.stop.latin import STOPS_LIST
from cltk.lemmatize.latin.backoff import BackoffLatinLemmatizer
import string


class Preprocessor:

    def __init__(self, tokens:list, remove_list:list, lemmatize = True, stops=STOPS_LIST):
        self.stops = stops
        self.tokens = tokens
        self.remove_list = remove_list
        self.lemmatize_bool = lemmatize

    def lemmatize(self):
        b_lemmatizer = BackoffLatinLemmatizer()
        res = b_lemmatizer.lemmatize(self.tokens)
        return_list = []
        for i in range(len(res)):
            return_list.append(res[i][1])

        return return_list

    def remove_digits(self):
        r = []
        for j in range(len(self.tokens)):
            word = self.tokens[j]
            r.append(''.join([i for i in word if not i.isdigit()]))

        return r

    def punctuation(self, word):
        exclude = string.punctuation
        exclude +=  '®'

        return ''.join([ch for ch in word if ch not in exclude])

    def numbers(self):
        inputs = ["i", "v", "x", "l", "c", "d", "m"]
        r = []
        for i in self.tokens:
            count = 0
            lenght = len(i)
            for j in i:
                if j in inputs:
                    count += 1
                
            if count != lenght:
                r.append(i)

        return r

    def remove_words_word_list(self):
        return_dict = {k:0 for k in self.tokens}

        for sub in self.remove_list:
            for t in self.tokens:
                if sub in t:
                    return_dict[t] +=1
        
        return [k for k,v in return_dict.items() if v == 0]


    def cleaning(self):
        print(f'Number of tokens pre-cleaning: {len(self.tokens)}')

        self.tokens = [i.lower().strip() for i in self.tokens]
        self.tokens = self.remove_digits()
        self.tokens = [self.punctuation(i) for i in self.tokens]
        self.tokens = self.numbers()
        self.tokens = self.remove_words_word_list()
        self.tokens = [i for i in self.tokens if i not in self.stops]
        self.tokens = [i for i in self.tokens if len(i) >= 2]

        if self.lemmatize_bool:
            self.tokens = self.lemmatize()
            
        print(f'Number of tokens post-cleaning: {len(self.tokens)}')        
        return self.tokens


         
