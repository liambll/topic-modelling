# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:18:46 2017

@author: linhb
"""

# pyspark --packages org.mongodb.spark:mongo-spark-connector_2.11:2.0.0
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, DoubleType, IntegerType
from pyspark.ml.clustering import LDA

my_spark = SparkSession \
    .builder \
    .appName("myApp") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/publications.papers") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/publications.papers") \
    .getOrCreate()

df = my_spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

df = df.drop('content')
df_IDF.select('features').head()

# clean up text
def extract_main_content(text):
    # Main Content are between "Abstract" and "References"
    text = text.lower()
    abstract_index = text.find("abstract")
    if (abstract_index == -1):
        abstract_index = 0
    reference_index = text.rfind("references")
    if (reference_index == -1):
        reference_index = len(text)
    return text[abstract_index:reference_index]
     
udf_extract_content = udf(lambda x: extract_main_content(x), StringType())
df1 = df.withColumn("text_main", udf_extract_content(df.text))

tokenizer = RegexTokenizer(inputCol="text_main", outputCol="words", pattern="\\P{Alpha}+")
df2 = tokenizer.transform(df1)

remover = StopWordsRemover(inputCol="words", outputCol="words2")
df3 = remover.transform(df2)

def remove_words(aList):
    stopWords = ['abstract','keyword','introduction','conclusion','acknowledgement']
    return [x for x in aList if (len(x)>1 and x not in stopWords)]

udf_remove_words = udf(lambda x: remove_words(x), ArrayType(StringType()))
df4 = df3.withColumn("words3", udf_remove_words(df3.words2))

# text to feature vector - TF_IDF
# should not use HashingTF because we need word-index dictionary
countTF = CountVectorizer(inputCol="words3", outputCol="raw_features", minTF=1.0, minDF=1.0)
countTF_model = countTF.fit(df4)
countTF.getVocabSize()
df_countTF = countTF_model.transform(df4)
vocab = countTF_model.vocabulary

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_countTF)
df_IDF = idf_model.transform(df_countTF)
df_IDF.cache()

# LDA Model
lda = LDA(k=2, seed=1, optimizer="online", maxIter=100, featuresCol='features')
lda_model = lda.fit(df_IDF)

# evaluation: high likelihood, low perplexity
lda_model.logPerplexity(df_IDF)
lda_model.logLikelihood(df_IDF)

# LDA model description
lda_model.vocabSize()
topics = lda_model.describeTopics(maxTermsPerTopic=10)

def lookup_words(termIndices, vocab):
    return [vocab[i] for i in termIndices]

udf_lookup_words = udf(lambda x: lookup_words(x, vocab), ArrayType(StringType()))
topics_words = topics.withColumn("words", udf_lookup_words(topics.termIndices))
topics_words.cache()
topics_words.head()

lda_model.topicsMatrix()
lda_model.estimatedDocConcentration()

# get topicDistribution





