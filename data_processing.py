# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:18:46 2017

@author: linhb
"""

from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTText

def pdf_to_text(pdfname):
    fp = open(pdfname, 'rb')
    parser = PDFParser(fp)
    doc = PDFDocument()
    parser.set_document(doc)
    doc.set_parser(parser)
    doc.initialize('')
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    # Process each page contained in the document.
    text = ""     
    for page in doc.get_pages():
        interpreter.process_page(page)
        layout = device.get_result()
        for lt_obj in layout:
            if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine) \
                    or isinstance(lt_obj, LTText):
                text = text + '\n' + lt_obj.get_text()

    fp.close()
    device.close()
    return text.replace('\n\n', '\n')

'''
# Test:
pdfname = "Project/14-249.pdf"
text = pdf_to_text(pdfname)
text = text.lower()
abstract_index = text.find("abstract")
keywords_index = text.find("keywords")
fulltext_index = text.find("1. introduction")
abstract = text[abstract_index: keywords_index]
keywords = text[keywords_index: fulltext_index]
'''