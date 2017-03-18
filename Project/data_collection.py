# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:18:46 2017

@author: linhb
"""
# Run without output to avoid connection reset when accessing cluster.
# /home/llbui/anaconda3/bin/python data_collection.py -s > /dev/null 2>&1

import os
from pymongo import MongoClient
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.selector import Selector
from scrapy.http import FormRequest
import hashlib
import data_processing

# SFU Username and Password in order to access some journals
# These variables should be stored in environment for security
USERNAME = 'your_username'
PASSWORD = 'your_password'

class PDF_File(scrapy.Item):
    source = scrapy.Field()
    title = scrapy.Field()
    authors = scrapy.Field()
    abstract = scrapy.Field()
    keywords = scrapy.Field()
    url = scrapy.Field()

# Process PDF and Save document to MongoDB
def processPDF(response):
    item = response.meta['item']
    documentid = int(hashlib.sha256(item['title'].encode('utf-8')).hexdigest(), 16) % 10**8

    # save pdf file with a temp filename in order to extract text
    path = str(documentid) + '.pdf'
    f = open(path, 'wb')
    f.write(response.body)
    f.close()
    # Process PDF file to extract text
    text = data_processing.pdf_to_text(path)
    os.remove(f.name)

    # Save Item object to MongoDB
    paper = {"_id": documentid,
             "source": item['source'],
             "title": item['title'],
             "authors": item['authors'],
             "abstract": item['abstract'],
             "keywords": item['keywords'],
             "url": item['url'],
             "content": response.body,
             "text": text
             }
    collection.save(paper)
        
################################################
# Web Crawler for Journal of Machine Learning 
################################################   
class Spider_JMLR(scrapy.Spider):
    name = "JMLR"

    def start_requests(self):
        urls = []
        # JMLR papers in the past 5 years
        for i in range(13,19,1):
            urls.append('http://www.jmlr.org/papers/v%i/' %i)
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parseMain)

    def parseMain(self, response):
        self.log('Processing %s' % response.url)
        dl_list = response.css('dl')
        for dl in dl_list:
            item = PDF_File()
            item['source'] = 'JRML'
            item['title'] = dl.css('dt::text').extract_first()
            item['authors'] = dl.css('i::text').extract_first()
            item['keywords'] = ''
            
            # pdf link
            pdf_url = dl.css('a::attr(href)').extract()[1]
            pdf_fullurl = response.urljoin(pdf_url)
            item['url'] = pdf_fullurl

            # abstract link
            abstract_url = dl.css('a::attr(href)').extract()[0]
            abstract_fullurl = response.urljoin(abstract_url)
            request = scrapy.Request(abstract_fullurl, callback=self.processPaper)
            request.meta['item'] = item
            yield request 
                  
    def processPaper(self, response):
        # get abstract text
        text = Selector(text=response.body).xpath('//text()').extract()
        text = ''.join(text)
        abstract_index = text.lower().find("abstract")
        abs_index = text.lower().find("[abs]")
        abstract = text[abstract_index+8: abs_index]
        abstract = abstract.replace("\n", " ").strip()
        item = response.meta['item']
        item['abstract'] = abstract

        # process pdf link
        request = scrapy.Request(item['url'], callback=processPDF)
        request.meta['item'] = item
        yield request

################################################
# Web Crawler for Neural Information Processing Systems
################################################
class Spider_NIPS(scrapy.Spider):
    name = "NIPS"

    def start_requests(self):
        urls = []
        # JMLR papers in the past 5 years
        for i in range(0,1,1):
            urls.append('http://papers.nips.cc/book/advances-in-neural-information-processing-systems-%i-%i' %(25+i, 2012+i))
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parseMain)

    def parseMain(self, response):
        self.log('Processing %s' % response.url)
        dl_list = response.css('li')
        for dl in dl_list[1:]: # skip the first li
            # process link to paper details
            url = dl.css('a::attr(href)').extract()[0]
            fullurl = response.urljoin(url)
            request = scrapy.Request(fullurl, callback=self.processPaper)
            yield request
                  
    def processPaper(self, response):
        sel = Selector(text=response.body)
        item = PDF_File()
        item['source'] = 'NIPS'
        item['title'] = sel.xpath('//meta[@name="citation_title"]/@content').extract_first()
        item['abstract'] = sel.xpath('//p[@class="abstract"]/text()').extract_first()
        item['authors'] = ', '.join(sel.xpath('//meta[@name="citation_author"]/@content').extract())
        item['keywords'] = ''
        item['url'] = sel.xpath('//meta[@name="citation_pdf_url"]/@content').extract_first()
        
        #process pdf link
        request = scrapy.Request(item['url'], callback=processPDF)
        request.meta['item'] = item
        yield request
        
################################################
# Web Crawler for SpringerLink Machine Learning
################################################
class Spider_SLML(scrapy.Spider):
    name = "SLML"

    def start_requests(self):
        urls = []
        # JMLR papers in the past 5 years
        for i in range(0,1,1):
            urls.append('https://link-springer-com.proxy.lib.sfu.ca/journal/10994/106/2/page/1')
        for url in urls:
            yield scrapy.Request(url=url, callback=self.check, dont_filter=True)

    def check(self, response):
        text = Selector(text=response.body).xpath('//text()').extract()
        if "Authentication Required" in text:
            print("Authentication Required!")
            return self.login(response)
        else:
            print("Authentication Done or Not Required!")
            return self.parseMain(response)

    def login(self, response):
        return FormRequest.from_response(response,
                    formdata={'user': USERNAME, 'pass': PASSWORD},
                    callback=self.check)
            
    def parseMain(self, response):
        self.log('Processing %s' % response.url)
        sel = Selector(text=response.body)
        dl_list = sel.xpath('//h3[@class="title"]//@href').extract()
        for url in dl_list[0:1]:
            # process link to paper details
            fullurl = response.urljoin(url)
            request = scrapy.Request(fullurl, callback=self.processPaper)
            yield request
                  
    def processPaper(self, response):
        sel = Selector(text=response.body)
        item = PDF_File()
        item['source'] = 'NIPS'
        item['title'] = sel.xpath('//meta[@name="citation_title"]/@content').extract_first()
        item['abstract'] = sel.xpath('//p[@id="Par1"]/text()').extract_first()
        item['authors'] = ', '.join(sel.xpath('//meta[@name="citation_author"]/@content').extract())
        item['keywords'] = ', '.join(sel.xpath('//span[@class="Keyword"]/text()').extract())
        item['url'] = sel.xpath('//meta[@name="citation_pdf_url"]/@content').extract_first()
        
        #process pdf link
        request = scrapy.Request(item['url'], callback=processPDF)
        request.meta['item'] = item
        yield request
 
################################################
# Web Crawler for SpringerLink Machine Learning
################################################
class Spider_IEEE(scrapy.Spider):
    name = "IEEE"
    
    #Jacob, please implement your class here
    # You can look at Spider_SLML as reference to handle authentication
 

################################################
# Set up MongoDB and Web Crawler
################################################

# Setup MongoDB Connection
# Start MongoDB Server: mongod.exe --dbpath D:\Training\Software\MongoDB\data
# export PATH=/home/llbui/mongodb/mongodb-linux-x86_64-3.4.2/bin:$PATH
# mongod --dbpath /home/llbui/mongodb/data
# mongo mongodb://gateway.sfucloud.ca:27017
client = MongoClient("mongodb://localhost:27017")
#client = MongoClient("mongodb://gateway.sfucloud.ca:27017")
db = client['publications']
collection = db['papers']

# Start Web Crawler
process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})
#process.crawl(Spider_JMLR)
process.crawl(Spider_NIPS)
#process.crawl(Spider_SLML)

process.start()

# Close MongoDB connection
client.close()
