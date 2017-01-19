# -*- coding: utf-8 -*-

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StopWordsRemover
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import sys
 
def main(argv=None):
    if argv is None:
        inputs_train = sys.argv[1]
        inputs_test = sys.argv[2]
    
    conf = SparkConf().setAppName('sentiment-analysis-tfidf')
    sc = SparkContext(conf=conf)
    sqlCt = SQLContext(sc)
    
    #read train json file and prepare data (label, feature)
    text = sqlCt.read.json(inputs_train)
    train = text.select('overall','reviewText').withColumnRenamed('overall','label')

    # ML PIPELINE:
    # Split at whitespace and characters that are not letter
    tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\P{Alpha}+")
    
    # stopword remover
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # TF-IDF Features
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    
    # linear Regression Model
    lr = LinearRegression(maxIter=20, regParam=0.1)
    # Final Pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
    
    # FIT MODEL USING CROSS VALIDATION
    # Parameter grid for cross validation: numFeatures and regParam
    paramGrid = ParamGridBuilder() \
        .addGrid(hashingTF.numFeatures, [100, 1000, 5000, 10000]) \
        .addGrid(lr.regParam, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) \
        .build()
        
    # 5-fold cross validation
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=5)
    
    # Run cross-validation, and choose the best set of parameters.
    model = crossval.fit(train)
    
    # EVALUATION
    evaluator = RegressionEvaluator()
    # RMSE on train data
    prediction_train = model.transform(train)
    rmse_train = evaluator.evaluate(prediction_train)

    #read test json file and prepare data (label, feature)
    text = sqlCt.read.json(inputs_test)
    test= text.select('overall','reviewText').withColumnRenamed('overall','label')
    
    # Evaluate the model on test data
    prediction_test = model.transform(test)
    rmse_test = evaluator.evaluate(prediction_test)
    
    # Print Result
    print("MODEL WITH TF_IDF features:")    
    print("Train RMSE: %f" % rmse_train)
    print("Test RMSE: %f" % rmse_test)

if __name__ == "__main__":
    main()