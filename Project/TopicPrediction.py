# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:18:46 2017

@author: linhb
"""

import sys, math
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizerModel, IDFModel
from pyspark.sql.functions import udf, desc
from pyspark.sql.types import StringType, ArrayType, FloatType
from pyspark.ml.clustering import LocalLDAModel

def main(argv=None):
    output_path = "/user/llbui/bigdata45_500"
    query = "treatment analysis using deep learning" #query string
    n = 10 #number of similar document to return
    feature = "abstract" #feature to compare
    if len(sys.argv) > 1:
        if len(sys.argv[1]) > 0:
            output_path = sys.argv[1]
    if len(sys.argv) > 2:
        if len(sys.argv[2]) > 0:
            query = sys.argv[2]
    if len(sys.argv) > 3:
        n = int(sys.argv[3])
      
    conf = SparkConf().setAppName('item-similarity-recommend')
    sc = SparkContext(conf=conf)
    sqlCt = SQLContext(sc)
    
    df = sc.parallelize([(0, query)]).toDF(["id", feature])
    
    tokenizer = RegexTokenizer(inputCol=feature, outputCol="words", pattern="\\P{Alpha}+")
    df2 = tokenizer.transform(df)
    
    remover = StopWordsRemover(inputCol="words", outputCol="words2")
    df3 = remover.transform(df2)
    
    udf_remove_words = udf(lambda x: remove_words(x), ArrayType(StringType()))
    df4 = df3.withColumn("words3", udf_remove_words(df3.words2))
    
    # text to feature vector - TF_IDF
    countTF_model = CountVectorizerModel.load(output_path + "/tf_model")
    df_countTF = countTF_model.transform(df4)
    
    idf_model = IDFModel.load(output_path + "/idf_model")
    df_IDF = idf_model.transform(df_countTF)
    
    # LDA Model
    lda_model = LocalLDAModel.load(output_path + "/lda_model")
    
    #output topics for document -> topicDistribution
    df_Feature = lda_model.transform(df_IDF)
    feature_vector = df_Feature.select("id", "topicDistribution").collect()[0][1]
    print("Feature Vector:", feature_vector)
    
    #Load existing document
    df_Document = sqlCt.read.load(output_path + "/topicDistribution.parquet")
    udf_cosineSimilarity = udf(lambda x_vector: cosineSimilarity(x_vector, feature_vector), FloatType())
    df_Similarity = df_Document.withColumn("similarity", udf_cosineSimilarity("topicDistribution"))
    df_Similarity_Sorted = df_Similarity.sort(desc("similarity"))
    print(df_Similarity_Sorted.limit(n).select("_id", "title", "url", "similarity").collect())

if __name__ == "__main__":
    main()

def remove_words(aList):
    stopWords = ['abstract','keyword','introduction','conclusion','acknowledgement']
    return [x for x in aList if (len(x)>1 and x not in stopWords)]

def cosineSimilarity(x, y):
    xy = sum([i*j for (i,j) in zip(x,y)])
    x2 = math.sqrt(sum([i*i for i in x]))
    y2 = math.sqrt(sum([j*j for j in y]))
    if (x2*y2 == 0):
      return float(0.0)
    else:
      return float(1.0*xy/(x2*y2))