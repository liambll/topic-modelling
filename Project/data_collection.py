# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:18:46 2017

@author: linhb
"""

import os
from pymongo import MongoClient
import scrapy
from scrapy.crawler import CrawlerProcess
import data_processing

class PDF_File(scrapy.Item):
    source = scrapy.Field()
    title = scrapy.Field()
    authors = scrapy.Field()
    abstract = scrapy.Field()
    keywords = scrapy.Field()
    content = scrapy.Field()
    
class Spider_JMLR(scrapy.Spider):
    name = "JMLR"

    def start_requests(self):
        urls = []
        # JMLR papers in the past 5 years
        for i in range(18,19,1):
            urls.append('http://www.jmlr.org/papers/v%i/' %i)
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        self.log('Processing %s' % response.url)
        dl_list = response.css('dl')
        for dl in dl_list:
            item = PDF_File()
            item['source'] = 'JRML'
            item['title'] = dl.css('dt::text').extract_first()
            item['authors'] = dl.css('i::text').extract_first()
            item['abstract'] = ''
            item['keywords'] = ''
            item['content'] = ''
            
            pdf_url = dl.css('a::attr(href)').extract()[1]
            pdf_fullurl = response.urljoin(pdf_url)
            request = scrapy.Request(pdf_fullurl, callback=self.processPDF)
            request.meta['item'] = item
            yield request   
            
    def processPDF(self, response):
        # save pdf file
        path = response.url.split('/')[-1]
        
        f = open(path, 'wb')
        f.write(response.body)
        f.close()
        # Process PDF file
        text = data_processing.pdf_to_text(path) #Something wrong here!
        os.remove(f.name)
    
        
        # Save Item object to MongoDB
        item = response.meta['item']
        paper = {"_id": item['source']+' '+item['title'],
                 "source": item['source'],
                 "title": item['title'],
                 "authors": item['authors'],
                 "abstract": item['abstract'],
                 "keywords": item['keywords'],
                 "content": response,
                 "text": text
                 }
        collection.insert_one(paper)
        
        

# Setup MongoDB Connection
# Start MongoDB Server: mongod.exe --dbpath D:\Training\Software\MongoDB\data
# export PATH=/home/llbui/mongodb/mongodb-linux-x86_64-3.4.2/bin:$PATH
# mongod --dbpath /home/llbui/mongodb/data
# mongo mongodb://gateway.sfucloud.ca:27017/publications
client = MongoClient("mongodb://localhost:27017")
db = client['publications']
collection = db['papers']

'''
f = open('14-249.pdf', 'rb')
content = f.read()
text = ''
f.close()

f2 = open('test.pdf', 'wb')
f2.write(content)
f2.close()

paper = {"_id": 'JRML'+' '+'a Title'
         "source": 'JRML',
         "title": 'a Title',
         "authors": 'list of authorszzzz',
         "abstract": 'an abstract',
         "keywords": 'list of keywordszzzz',
         "content": content,
         "text": text
         }
collection.insert_one(paper)
'''

# Start Web Crawler
process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})
process.crawl(Spider_JMLR)
process.start()

# Close MongoDB connection
client.close()