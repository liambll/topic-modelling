# Big Data Analytics Project: Topic Model on Machine Learning Research Publications #
- web scraping function to retrieval publications from major machine learning journals & conferences
- topic modeling (Latent Dirichlet Allocation) to explore underlying topics
- graph analytics to understand and visualize relationship among publication contents
- a web-based demo for similarity-based publication search
- Technologies: Scala, Spark ML, Spark GraphX, MongoDB, Gephi, Django

I) Data Collection and Text Extraction:
=================================================================================================================
1) Requirement:
- Python 3
- pdfminer3k
- scrapy
- pymongo

2) Description:
- data_processing.py: process PDF file to extract text
- data_collection.py: contains 5 Scrappy objects to perform web-scraping from 5 machine learning research journals and store information in MongoDB

3) Running Instruction:
- Start MongoDB if needed: /home/llbui/mongodb/mongodb-linux-x86_64-3.4.2/bin/mongod /home/llbui/mongodb/data
The above MongoDB server should have "publications" database and "papers" collection.
- Run data_collection.py. If needed, run each crawler object seperately to utilize parallelism.

II) Data Exploration
=================================================================================================================
1) Requirement:
- wordcloud

2) Description:
- world_cloud/get_words.py: retrieve different document attributes (title, abstarct, text) from MongoDB
- world_cloud/word_cloud_test.py: visualize word frequencies using wordcloud

3) Running Instruction:
- Start MongoDB if needed
- Run get_words.py to retrieve text and then run word_cloud_test.py to visualize

III) Data Analysis
=================================================================================================================
1) Requirement
- Spark 2.0
- Standford NLP (for text stemming and lemmentization): The required jar file is located at /user/llbui/stanford-corenlp-3.4.1-models.jar

2) Description:
- TopicModel.scala: perform text pre-processing and topic model. Best model and final output(topic distributions for each document) are saved on Hadoop cluster at /user/llbui/bigdata45_500
- TopicDistribution.scala: read final output(topic distributions for each document) from parquet files and save to MongoDB
- TopicRelationship.scala: compute Chi-square score and construct graph to network analysis
- topics.csv: final identified topic ID and topic names. We look at major words contributing the each topic in order to decide topic names
- BigData.gephi: final network showing relationship among topics. We use Gephi to analyze Eigen Centrality and find interesting cluster/link.
- explore_topics.py: contains ad-hoc analysis and visualization tasks (eg. word importance for each topic using wordcloud, topic distribution over all the whole corpus, extract graphs from Spark RDD collect and convert to csv format readable by Gephi)

3) Running Instruction:
- Start MongoDB if needed. It is required for TopicModel.scala and TopicDistribution.scala to run.
- All Scala files are compiled in big-data-analytics-project_2.11-1.0.jar
spark-submit --master yarn --deploy-mode cluster --driver-memory 10g --executor-memory=5g --packages edu.stanford.nlp:stanford-corenlp:3.4.1,org.mongodb.spark:mongo-spark-connector_2.11:2.0.0 --jars stanford-corenlp-3.4.1-models.jar --class TopicModel big-data-analytics-project_2.11-1.0.jar "/user/llbui/bigdata45_500" "abstract" "online" 500 40 45
spark-submit --master yarn --deploy-mode client --driver-memory 10g --class TopicDistribution big-data-analytics-project_2.11-1.0.jar "/user/llbui/bigdata45_500" 
spark-submit --master yarn --deploy-mode client --driver-memory 10g --class TopicRelationship big-data-analytics-project_2.11-1.0.jar "/user/llbui/bigdata45_500" 0.045 

IV) DATA PRODUCT DEMO:
=================================================================================================================
1) Requirement:
- Django
- Celery

2) Description:
- web\LDA\topic_modeling\: contains web front-end files to enable user browsing document by topic and searching document by query
- web\LDA\spark_celery\: call Spark jobs in the back-end

3) Running Instruction:
- Start MongoDB server if needed
- Start Djiango server: Navigate to web\LDA\ folder and run python manage.py runserver
- Browse to http://127.0.0.1:8000/topic_modeling/index




