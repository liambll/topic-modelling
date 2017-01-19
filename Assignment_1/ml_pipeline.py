# -*- coding: utf-8 -*-
# USAGE: spark-submit --master yarn-client ml_pipeline.py
# The parquet data need to be available on cluster /user/*userId*/
# Outputs will be printed out starting with a line "FINAL RESULT:"

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

conf = SparkConf().setAppName("MLPipeline")
sc = SparkContext(conf=conf)

# Read training data as a DataFrame
sqlCt = SQLContext(sc)
trainDF = sqlCt.read.parquet("20news_train.parquet")
trainDF.cache() # to be used again for model with cross validation

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features", numFeatures=1000)
lr = LogisticRegression(maxIter=20, regParam=0.1)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training data.
model = pipeline.fit(trainDF)

# Evaluate the model on testing data
testDF = sqlCt.read.parquet("20news_test.parquet")
testDF.cache() # to be used again for model with cross validation
prediction = model.transform(testDF)
evaluator = BinaryClassificationEvaluator()
areaUnderROC = evaluator.evaluate(prediction)

# MODEL SELECTION WITH CROSS VALIDATION
# Parameter grid for cross validation: numFeatures and regParam
paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [1000, 5000, 10000]) \
    .addGrid(lr.regParam, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) \
    .build()

# 2-fold cross validation
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2)
    
# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(trainDF)

# Evaluate the model with cross validation on testing data
prediction = cvModel.transform(testDF)
evaluator = BinaryClassificationEvaluator()
areaUnderROC_cv = evaluator.evaluate(prediction)

# Print Result
print('FINAL RESULT:')
print('- areaUnderROC with numFeatures 1000 and regParam 0.1: ' + str(areaUnderROC))
print('- areaUnderROC with cross validation tuning: ' + str(areaUnderROC_cv))