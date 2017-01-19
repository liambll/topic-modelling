# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec, RegexTokenizer, StopWordsRemover, CountVectorizer, Normalizer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.clustering import KMeans
import sys

def find_cluster(sentence, dictionary):
    clusters = []
    for word in sentence:
        if word in dictionary.value:
            clusters.append(dictionary.value[word])
    return clusters
    
def main(argv=None):
    if argv is None:
        inputs_train = sys.argv[1]
        inputs_test = sys.argv[2]
    
    conf = SparkConf().setAppName('sentiment-analysis-word2vec-cluster')
    sc = SparkContext(conf=conf)
    sqlCt = SQLContext(sc)
    
    #read train json file and prepare data (label, feature)
    text = sqlCt.read.json(inputs_train)
    train = text.select('overall','reviewText').withColumnRenamed('overall','label')

    # Split at whitespace and characters that are not letter
    tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\P{Alpha}+")
    data_tokenizer = tokenizer.transform(train)
    
    # stopword remover
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    data_remover = remover.transform(data_tokenizer)
    data_remover.cache()
    
    ## SET UP DICTIONARY WORD-CLUSTERS:
    # Word2Vec Features
    word2Vec = Word2Vec(inputCol="filtered_words", outputCol="word2vec_features")
    model_word2Vec = word2Vec.fit(data_remover)
    vocabulary = model_word2Vec.getVectors()
    vocabulary.cache()
    
    # Trains a k-means model.
    kmeans = KMeans(featuresCol="vector", predictionCol="cluster").setK(100).setSeed(1)
    model_kmeans = kmeans.fit(vocabulary)
    data_vocabulary = model_kmeans.transform(vocabulary)
    
    # Create and broadcast the dictionary <word, cluster>
    rdd_dictionary = data_vocabulary.select('word','cluster') \
        .map(lambda row: (row['word'],str(row['cluster'])))
    dictionary = dict(rdd_dictionary.collect())
    dictionary_broadcast = sc.broadcast(dictionary)
    
    ## ML PIPELINE
    # convert words to corresponding clusters
    train_cluster = data_remover \
        .map(lambda row: (row['label'], find_cluster(row['filtered_words'], dictionary_broadcast))) \
        .toDF(['label','clusters'])
        
    # feature  based on clusters
    count_vectorizer = CountVectorizer(inputCol="clusters", outputCol="count")
    normalizer = Normalizer(inputCol="count", outputCol="features", p=1.0)                     
    
    # linear Regression Model
    lr = LinearRegression(maxIter=20, regParam=0.1)
    
    # Final Pipeline
    pipeline = Pipeline(stages=[count_vectorizer, normalizer, lr])
    
    # FIT MODEL USING CROSS VALIDATION
    # Parameter grid for cross validation: numFeatures and regParam
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) \
        .build()
        
    # 5-fold cross validation
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=5)
    
    # Run cross-validation, and choose the best set of parameters.
    model = crossval.fit(train_cluster)

    # EVALUATION
    evaluator = RegressionEvaluator()
    # RMSE on train data
    prediction_train = model.transform(train_cluster)
    rmse_train = evaluator.evaluate(prediction_train)

    #read test json file and prepare data (label, feature)
    text = sqlCt.read.json(inputs_test)
    test= text.select('overall','reviewText').withColumnRenamed('overall','label')
    data_tokenizer = tokenizer.transform(test)
    data_remover = remover.transform(data_tokenizer)
    test_cluster = data_remover \
        .map(lambda row: (row['label'], find_cluster(row['filtered_words'], dictionary_broadcast))) \
        .toDF(['label','clusters'])
    
    # Evaluate the model on test data
    prediction_test = model.transform(test_cluster)
    rmse_test = evaluator.evaluate(prediction_test)
 
    # Print Result
    print("MODEL WITH Word Clustering features:")
    print("Train RMSE: %f" % rmse_train)
    print("Test RMSE: %f" % rmse_test)

if __name__ == "__main__":
    main()
        