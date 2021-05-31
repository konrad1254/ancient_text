import pytesseract
from pdf2image import convert_from_path
import glob
import os
import re

def magic_converter(document, path_to_save_text):

    os.chdir(str(path_to_save_text))

    pdfs = glob.glob(document)

    for pdf_path in pdfs:
        pages = convert_from_path(pdf_path, 500)

        for pageNum,imgBlob in enumerate(pages):
            text = pytesseract.image_to_string(imgBlob,lang='eng')

            with open(f'{pdf_path[:-4]}_page{pageNum}.txt', 'w') as the_file:
                the_file.write(text)

