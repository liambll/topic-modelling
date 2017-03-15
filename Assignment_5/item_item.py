# -*- coding: utf-8 -*-
# USAGE: spark-submit --master yarn-client --num-executors 10 item_item_CF.py /user/llbui/a5 plot
# First argument is a directory containing input dataset with:
    # MovieLens100K_train.txt
    # MovieLens100K_test.txt
# Second argument "plot" is optional to show test RMSE plot if run locally

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as func
import sys

def main(argv=None):
    plot = ""
    if argv is None:
        inputs = sys.argv[1]
            
    conf = SparkConf().setAppName('item-similarity-recommend')
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

    # Average rating per user
    average_user_rating = train.groupBy("userID").agg(func.mean("rating").alias("avg_rating"))
    user_rating = train.join(average_user_rating, on="userID")
    user_rating_dev = user_rating.withColumn("dev_rating", \
                    user_rating["rating"] - user_rating["avg_rating"]) \
                    .withColumnRenamed("movieID", "movieID_dev") \
                    .select("userID","movieID_dev", "dev_rating", "avg_rating")
    user_rating_dev.cache()
    
    # compute item-item similarities by joining ratings from the same user
    # with movieID < movieID2 to avoid duplicated pairs
    train2 = train.withColumnRenamed("movieID", "movieID2") \
                    .withColumnRenamed("rating", "rating2")
    itempairs = train.join(train2, on=((train.userID == train2.userID) & \
                                       (train.movieID < train2.movieID2)))
    itempairs_group = itempairs.groupBy("movieID", "movieID2").agg( \
        func.corr("rating","rating2").alias("correlation"))
    itempairs_similarity = itempairs_group.filter("correlation is not Null") \
            .select("movieID", "movieID2", "correlation") \
            .withColumnRenamed("movieID", "movieID1")
    itempairs_similarity.cache()           
    
    # List to store results:
    model_result = []
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        itempairs_filtered = itempairs_similarity.filter("correlation >= " + str(threshold))
        
        # join test data with ratings from the same users, movieID1 < movieID2
        smallerID = func.udf(lambda id1, id2: id1 if id1 < id2 else id2, IntegerType())
        greaterID = func.udf(lambda id1, id2: id1 if id1 >= id2 else id2, IntegerType())
        test_user_rating = test.join(user_rating_dev, "userID") \
                .withColumn("movieID1", smallerID("movieID","movieID_dev")) \
                .withColumn("movieID2", greaterID("movieID","movieID_dev"))
        
        # join with itempairs_filtered to select only similar items rated by the same user
        test_user_rating_derived = test_user_rating.join(itempairs_filtered, ["movieID1","movieID2"]) \
                        .select("userID", "movieID", "rating", \
                                "correlation", "dev_rating", "avg_rating")
        
        # make prediction
        test_user_rating_group = test_user_rating_derived.groupBy("userID", "movieID", "rating", "avg_rating") \
                    .agg((func.sum(test_user_rating_derived.dev_rating*test_user_rating_derived.correlation) \
                        / func.sum(test_user_rating_derived.correlation)).alias("prediction_dev"))
        prediction_test = test_user_rating_group.withColumn("prediction", \
                        test_user_rating_group.avg_rating + test_user_rating_group.prediction_dev) \
                        .select("userID", "movieID", "rating", "prediction")
        
        # RMSE on test data - excluding users who do not have any prediction
        rmse_test = evaluator.evaluate(prediction_test)
        model_result.append((threshold, rmse_test))
        
    # Show plot if run locally
    if (plot == "plot"):
        plotRMSE(model_result)
    
    # Print results
    print("ITEM SIMILARITY COLLABORATIVE FILTERING: ")
    for i in model_result: 
        print("- Threshold = %3.1f: Test RMSE = %s" %(i[0], i[1]))


if __name__ == "__main__":
    main()

def plotRMSE(model_result):
    import matplotlib.pyplot as plt
    import numpy as np

    x = [i[0] for i in model_result]
    y = [i[1] for i in model_result]
    
    # Plot Bar Chart
    fig, ax = plt.subplots()
    ind = np.arange(len(x))
    rects = ax.bar(ind, y, 0.5, color='b')
    
    ax.set_title('Fit Item-Item Based Recommendation')
    ax.set_ylabel('Test RMSE')
    ax.set_ylim([min(y)-0.01,max(y)+0.01])
    ax.set_xlabel('Threshold')
    ax.set_xticks(ind + 0.5 / 2)
    ax.set_xticklabels(x)
    
    for i in np.arange(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%10.4f' % y[i],
                ha='center', va='bottom')