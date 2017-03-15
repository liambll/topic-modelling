# -*- coding: utf-8 -*-
# USAGE: spark-submit --master yarn-client --num-executors 10 anomaly_detection.py /user/llbui/a3/logs-features 8 0.97
# First argument is a directory containing parquet input. The code will use toy dataset if no argument is provided.
# Second and third argument is k and threshold. If these arguments are not provided, default k=8 and threshold = 0.97 will be used.

from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, StringType
from pyspark.sql.functions import udf
import sys

from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
 
class AnomalyDetection():

    def readToyData(self):
        data = [(0, ["http", "udt", 0.4]), \
                (1, ["http", "udf", 0.5]), \
                (2, ["http", "tcp", 0.5]), \
                (3, ["ftp", "icmp", 0.1]), \
                (4, ["http", "tcp", 0.4])]
        schema = ["id", "rawFeatures"]
        self.rawDF = sqlCt.createDataFrame(data, schema)
        
    def readData(self, filename):
        self.rawDF = sqlCt.read.parquet(filename).cache()

    def cat2Num(self, df, indices):
        """
            Write your code!
        """
        # function to select one feature from a list of feature
        def select_feature(raw_feature, index):
            return raw_feature[index]

        # function to select remove features from a list of feature
        def delete_feature(raw_feature, indices):
            feature = [i for j, i in enumerate(raw_feature) if j not in indices]
            return Vectors.dense(feature)

        # Get categorical features and perform One-Hot Encoding
        df_prev = df
        for index in indices:
            select_feature_udf = udf(lambda x: select_feature(x, index), StringType())
            df_encoded = df_prev.withColumn("cat_"+str(index), select_feature_udf("rawFeatures"))
            # string index
            stringIndexer = StringIndexer(inputCol="cat_"+str(index), outputCol="cat_index_"+str(index))
            model_stringIndexer = stringIndexer.fit(df_encoded)
            indexed = model_stringIndexer.transform(df_encoded)

            # one-hot encode
            encoder = OneHotEncoder(inputCol="cat_index_"+str(index), outputCol="cat_vector_"+str(index), dropLast=False)
            encoded = encoder.transform(indexed)
            df_prev = encoded

        # Get continious features by removing categorical indices from rawFeatures
        delete_feature_udf = udf(lambda x: delete_feature(x, indices), VectorUDT())
        df_cont = df_prev.withColumn("cont", delete_feature_udf("rawFeatures"))
        
        # Combine one-hot encoded categorical and continious features
        feature = []
        for index in indices:
            feature.append("cat_vector_"+str(index))
        feature.append("cont")
        assembler = VectorAssembler( inputCols=feature, outputCol="features")
        df_transformed = assembler.transform(df_cont) \
            .select("id","rawFeatures","features")
        return df_transformed

    def addScore(self, df):
        """
            Write your code for Score function
        """
        
        # score function
        def score(x, max_size, min_size):
            return 1.0*(max_size-x)/(max_size-min_size)
        
        # get min and max cluster size
        cluster_size = df.groupBy("prediction").count()
        max_min = cluster_size.agg(max("count"),min("count")).collect()[0]
        max_size = max_min["max(count)"]
        min_size = max_min["min(count)"]
        
        # get score
        scoreUDF = udf(lambda x: score(x, max_size, min_size), FloatType())
        cluster_score = cluster_size.withColumn("score", scoreUDF("count"))
        df_score = df.join(cluster_score, "prediction") \
            .select("id","rawFeatures","features","prediction","score")
        return df_score
    
    
    def detect(self, k, t):
        #Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show()

        #Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)

        #Adding the prediction column to df1
        modelBC = sc.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show()

        #Adding the score column to df2; The higher the score, the more likely it is an anomaly 
        df3 = self.addScore(df2).cache()
        df3.show()    

        return df3.where(df3.score > t)


if __name__ == "__main__":
    # initialize Spark Context
    conf = SparkConf().setAppName('abnomaly-detection')
    sc = SparkContext(conf=conf)
    sqlCt = SQLContext(sc)

    ad = AnomalyDetection()
    # read input if file is provided. If not, load toy data
    if len(sys.argv) > 1:
        inputs = sys.argv[1]
        ad.readData(inputs)
    else:
        ad.readToyData()
    
    #load k and threshold if provided
    if len(sys.argv) > 3:
        k = int(sys.argv[2])
        threshold = float(sys.argv[3])
    else:
        k = 8
        threshold = 0.97

    anomalies = ad.detect(k, threshold)
    anomalies.show()
    print("Number of anomalies: %i " % anomalies.count())