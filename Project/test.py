# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:18:46 2017

@author: linhb
"""

from scrapy.selector import Selector
from scrapy.http import HtmlResponse

f = open('Project/test.html', 'r')
body = f.read()
f.close()
body = '<html><body><span>good</span></body></html>'
response = HtmlResponse(url='http://example.com', body=body, encoding='utf-8')
text = Selector(text=body).xpath('//text()').extract()
text = ''.join(text)
abstract_index = text.lower().find("abstract")
abs_index = text.lower().find("[abs]")
abstract = text[abstract_index+8: abs_index]
abstract = abstract.replace("\n", " ").strip()

keywords = text[keywords_index: fulltext_index]