import pytesseract
from pdf2image import convert_from_path
import glob
import os
import re
from ancient_text import utils

def magic_converter(document, path_to_save_text):

    os.chdir(str(path_to_save_text))

    pdfs = glob.glob(document)

    for pdf_path in pdfs:
        pages = convert_from_path(pdf_path, 500)

        for pageNum,imgBlob in enumerate(pages):
            text = pytesseract.image_to_string(imgBlob,lang='eng')

            with open(f'{pdf_path[:-4]}_page{pageNum}.txt', 'w') as the_file:
                the_file.write(text)


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def load_text(path):
	files = utils.find_txt_filenames(path)
	sorted_files = sorted(files, key = natural_keys)
	data = utils.string_conversion(sorted_files)
	return data
