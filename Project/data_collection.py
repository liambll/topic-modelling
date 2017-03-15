# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:18:46 2017

@author: linhb
"""

import scrapy
from scrapy.crawler import CrawlerProcess

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'http://www.jmlr.org/papers/v18/'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        dl_list = response.css('dl')
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'w+', encoding="utf-8") as f:
            for item in dl_list:
                f.write(item.css('dt::text').extract_first())
                f.write("\n")
                f.write(item.css('i::text').extract_first())
                f.write("\n")
                pdf_url = item.css('a::attr(href)').extract()[1]
                pdf_fullurl = response.urljoin(pdf_url)
                f.write(pdf_fullurl)
                f.write("\n")
                yield scrapy.Request(pdf_fullurl, callback=self.processPDF)           
        self.log('Saved file %s' % filename)
        
    def processPDF(self, response):
        path = response.url.split('/')[-1]
        with open(path, 'wb') as f:
            f.write(response.body)

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})
process.crawl(QuotesSpider)
process.start()