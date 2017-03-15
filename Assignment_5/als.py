# -*- coding: utf-8 -*-
# USAGE: spark-submit --master yarn-client --num-executors 10 matrix_factor_CF.py /user/llbui/a5 plot
# First argument is a directory containing input dataset with:
    # MovieLens100K_train.txt
    # MovieLens100K_test.txt
    # u.item
# Second argument "plot" is optional to show test RMSE plot if run locally

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.clustering import KMeans
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import sys

def main(argv=None):
    plot = ""
    if argv is None:
        inputs = sys.argv[1]
        if (len(sys.argv) > 2):
            plot = sys.argv[2] # "plot" to show test RMSE plot
            
    conf = SparkConf().setAppName('matrix-factorization-recommend')
    sc = SparkContext(conf=conf)
    sqlCt = SQLContext(sc)
    
    #read train text file and prepare rating data (userID, movieID, rating)
    text = sqlCt.read.text(inputs+"/MovieLens100K_train.txt")
    train = text.map(lambda row: row.value.split("\t")) \
                .map(lambda l: (int(l[0]), int(l[1]), float(l[2]))) \
                .toDF(["userID", "movieID", "rating"])
    train.cache()
    
    #read test text file and prepare rating data (userID, movieID, rating)
    text = sqlCt.read.text(inputs+"/MovieLens100K_test.txt")
    test = text.map(lambda row: row.value.split("\t")) \
                .map(lambda l: (int(l[0]), int(l[1]), float(l[2]))) \
                .toDF(["userID", "movieID", "rating"])
    test.cache()
    
    #read movie names
    text = sqlCt.read.text(inputs+"/u.item")
    movie_names = text.map(lambda row: row.value.split("|")) \
                .map(lambda l: (int(l[0]), l[1])) \
                .toDF(["id", "movieName"])
    movie_names.cache()
    
    # Build the recommendation model using explicit ALS
    als = ALS(maxIter=20, userCol="userID", itemCol="movieID", ratingCol="rating")
  
    # List to store results:
    model_result = []
    cluster_result = []
        
    # Parameter grid for cross validation
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    ranks = [2, 4, 8, 16, 32, 64, 128, 256]
    for rank in ranks:
        paramGrid = ParamGridBuilder() \
        .addGrid(als.rank, [rank]) \
        .build()
        
        # 5-fold cross validation
        crossval = CrossValidator(estimator=als,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=5)
    
        # Run cross-validation.
        model = crossval.fit(train)

        # RMSE on test data - filtering out new users who would not have any prediction
        prediction_test = model.transform(test).filter("prediction <> 'NaN'")
        rmse_test = evaluator.evaluate(prediction_test)
        model_result.append((rank, rmse_test))
    
        # K-mean clustering for items based on 50 factors            
        item_factors = model.bestModel.itemFactors \
            .withColumn("features_vector", udf(lambda x: Vectors.dense(x),VectorUDT())("features")) \
            .cache()
        kmeans = KMeans(featuresCol="features_vector", predictionCol="cluster", \
                        initMode="random", k=50, seed = 1)
        model_kmeans = kmeans.fit(item_factors)
        item_clusters = model_kmeans.transform(item_factors)
        item_factors.unpersist()
        
        # Number of items small enough to collect
        two_clusters = item_clusters.filter("cluster < 2")   \
                .join(movie_names, on="id") \
                .select("cluster", "movieName") \
                .map(lambda row: (row[0],row[1])).collect()
        cluster1 = list(map(lambda x: x[1].encode("utf-8"),
                       filter(lambda x: x[0]==0, two_clusters)))
        cluster2 = list(map(lambda x: x[1].encode("utf-8"),
                       filter(lambda x: x[0]==1, two_clusters)))
        cluster_result.append((rank, (cluster1, cluster2)))
        
    # Show plot if run locally
    if (plot == "plot"):
        plotRMSE(model_result)
    
    # Print results
    print("MATRIX FACTORIZATION COLLABORATIVE FILTERING: ")
    for i in model_result: 
        print("- Rank = %i: Test RMSE = %s" %(i[0], i[1]))
    
    print("\nTwo Clusters: ")
    for i in cluster_result: 
        print("- Rank = %i:\n   Cluster-1: %s\n   Cluster-2: %s" \
              %(i[0], i[1][0], i[1][1]))

if __name__ == "__main__":
    main()

# Function to plot test RMSE
def plotRMSE(model_result):
    import matplotlib.pyplot as plt
    import numpy as np
    x = [i[0] for i in model_result]
    y = [i[1] for i in model_result]
    
    # Plot Bar Chart
    fig, ax = plt.subplots()
    ind = np.arange(len(x))
    rects = ax.bar(ind, y, 0.5, color='b')
    
    ax.set_title('Fit ALS with 5-fold cross validation and max 20 iterations')
    ax.set_ylabel('Test RMSE')
    ax.set_ylim([min(y)-0.01,max(y)+0.01])
    ax.set_xlabel('Rank')
    ax.set_xticks(ind + 0.5 / 2)
    ax.set_xticklabels(x)
    
    for i in np.arange(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%10.4f' % y[i],
                ha='center', va='bottom')