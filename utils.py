## needs to change according to incoming data
from cltk.corpus.utils.importer import CorpusImporter
import os
import re
import pycld2 as cld2

def find_txt_filenames( path_to_dir, suffix=".txt"):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix )]

def get_corpora(get_cltk_corpora = True):
    """
    Function to get the relevant downloads
    """
    home_path = os.path.expanduser("~")
    os.chdir(home_path)

    if get_cltk_corpora == True:
        my_latin_downloader = CorpusImporter('latin')
        print(f"Possible download: {my_latin_downloader.list_corpora}")
        print('------------')
        print('currently downloading: latin_text_latin_library and latin_models_cltk')

        my_latin_downloader.import_corpus('latin_text_latin_library')
        my_latin_downloader.import_corpus('latin_models_cltk')

def string_conversion(location_of_txt_list):
    return_list = []
    for file in location_of_txt_list:
            f = open(file, 'r')
            string = f.read()
            return_list.append(string)
    return ' '.join(return_list)

def return_language_detection(string):
    isReliable, textBytesFound, details, vectors = cld2.detect(
        string, returnVectors=True)
    return vectors

def language_extraction(string_input, language):
    return_list = []

    r = return_language_detection(string_input)

    for i in range(len(r)):
        if r[i][2] == 'LATIN':
            start = r[i][0]
            end = start + r[i][1]
            return_list.append(string_input[start:end])
    
    return ' '.join(return_list)


