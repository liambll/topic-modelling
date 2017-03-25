# Requirements:
1. Data Collection and Text Extraction
- Python 3
- pdfminer3k
- scrapy
- pymongo

Change the year 17-12
Load data:
/home/yiles/anaconda3/bin/python data_collection.py -s > /dev/null 2>&1

Check data loaded in:
/home/yiles/mongodb/mongodb-linux-x86_64-3.4.2/bin/mongo
use publications
db.papers.find({source: "ARXIV"}).count()
exit

2. ML Pipeline and Analytics
- Spark 2.0
- MongoDB Connector
