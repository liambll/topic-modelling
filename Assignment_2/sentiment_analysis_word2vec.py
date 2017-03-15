# -*- coding: utf-8 -*-
# USAGE: spark-submit --master yarn-client --num-executors 10 sentiment_analysis_word2vec.py /user/llbui/a2 /user/llbui/a2-test
# First argument is a directory containing train dataset
# Second argument is a directory containing test dataset

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec, RegexTokenizer, StopWordsRemover
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import sys
        
def main(argv=None):
    if argv is None:
        inputs_train = sys.argv[1]
        inputs_test = sys.argv[2]
    
    conf = SparkConf().setAppName('sentiment-analysis-word2vec')
    sc = SparkContext(conf=conf)
    sqlCt = SQLContext(sc)
    
    #read train json file and prepare data (label, feature)
    text = sqlCt.read.json(inputs_train)
    train = text.select('overall','reviewText').withColumnRenamed('overall','label')
    train.cache()
    
    ## DATA PROCESSING PIPELINE
    # Split at whitespace and characters that are not letter
    tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\P{Alpha}+")
    
    # stopword remover
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # Word2Vec Features - default: vector length 100
    word2Vec = Word2Vec(inputCol="filtered_words", outputCol="features")

    pipeline_data_processing = Pipeline(stages=[tokenizer, remover, word2Vec])
    model_data_processing = pipeline_data_processing.fit(train)
    train_processed = model_data_processing.transform(train)
    train.unpersist()
    train_processed.cache()
    
    ## ML PIPELINE
    # linear Regression Model
    lr = LinearRegression(maxIter=20, regParam=0.1)
    
    # FIT MODEL USING CROSS VALIDATION
    # Parameter grid for cross validation: numFeatures and regParam
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1, 1.0]) \
        .build()
        
    # 5-fold cross validation
    evaluator = RegressionEvaluator(metricName="rmse")
    crossval = CrossValidator(estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=5)
    
    # Run cross-validation, and choose the best set of parameters.
    model = crossval.fit(train_processed)

    # RMSE on train data
    prediction_train = model.transform(train_processed)
    rmse_train = evaluator.evaluate(prediction_train)
    train_processed.unpersist()
    
    ## EVALUATION ON TEST DATA
    #read test json file and prepare data (label, feature)
    text = sqlCt.read.json(inputs_test)
    test= text.select('overall','reviewText').withColumnRenamed('overall','label')
    test_processed = model_data_processing.transform(test)
    
    # Evaluate the model on test data
    prediction_test = model.transform(test_processed)
    rmse_test = evaluator.evaluate(prediction_test)
    
    # Print Result 
    result = "MODEL WITH Word2Vec features:\n"
    result = result + "-Train RMSE: " + str(rmse_train) + "\n"
    result = result + "-Test RMSE: " + str(rmse_test) + "\n"
    print(result)

if __name__ == "__main__":
    main()

