# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:18:46 2017

@author: linhb
"""
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

from io import StringIO

def pdf_to_text(pdfname):
    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    
    fp = open(pdfname, 'rb')
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()

    return text

'''
pdfname = "http://www.jmlr.org/papers/volume18/14-249/14-249.pdf"
text = pdf_to_text(pdfname)
text = text.lower()
abstract_index = text.find("abstract")
keywords_index = text.find("keywords")
fulltext_index = text.find("1. introduction")
abstract = text[abstract_index: keywords_index]
keywords = text[keywords_index: fulltext_index]
'''