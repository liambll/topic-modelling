# -*- coding: utf-8 -*-
# USAGE: spark-submit --master yarn-client --num-executors 10 sentiment_analysis_cluster.py /user/llbui/a2 /user/llbui/a2-test
# First argument is a directory containing train dataset
# Second argument is a directory containing test dataset

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline, Estimator, Model
from pyspark.ml.feature import Word2Vec, RegexTokenizer, StopWordsRemover, CountVectorizer, Normalizer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.clustering import KMeans
from pyspark.ml.util import keyword_only
from pyspark.ml.param.shared import HasInputCol, HasPredictionCol, Param
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import sys

        
# Custom Estimator to get word-cluster dictionary
class WordCluster(Estimator, HasInputCol, HasPredictionCol):
    @keyword_only
    def __init__(self, inputCol=None, predictionCol=None, k=None, vocabulary=None):
        super(WordCluster, self).__init__()
        self.k = Param(self, "k", "")
        self._setDefault(k=2)
        self.vocabulary = Param(self, "vocabulary", "")
        self._setDefault(vocabulary="")
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, predictionCol=None, k=None, vocabulary=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setK(self, value):
        self._paramMap[self.k] = value
        return self
        
    def getK(self):
        return self.getOrDefault(self.k)

    def setVocabulary(self, value):
        self._paramMap[self.vocabulary] = value
        return self

    def getVocabulary(self):
        return self.getOrDefault(self.vocabulary)

        
    def _fit(self, dataset):
        k = self.getK()
        vocabulary = self.getVocabulary()

        # Trains a k-means model on word-vectors
        kmeans = KMeans(featuresCol="vector", predictionCol="cluster", initMode="random") \
                .setK(k).setSeed(1)
        model_kmeans = kmeans.fit(vocabulary)
        data_vocabulary = model_kmeans.transform(vocabulary)
    
        # Create and broadcast the dictionary <word, cluster>
        rdd_dictionary = data_vocabulary.select('word','cluster') \
            .map(lambda row: (row['word'],str(row['cluster'])))
        dictionary = dict(rdd_dictionary.collect())      

        return (WordClusterModel()
            .setInputCol(self.getInputCol())
            .setPredictionCol(self.getPredictionCol())
            .setDictionary(dictionary)
            .setK(k))
 
# Custom Modelto use word-cluster dictionary to map vector of words to vector of clusters
class WordClusterModel(Model, HasInputCol, HasPredictionCol):
    @keyword_only
    def __init__(self, inputCol=None, predictionCol=None, dictionary=None, k=None):
        super(WordClusterModel, self).__init__()
        self.dictionary = Param(self, "dictionary", "")
        self._setDefault(dictionary="")
        self.k = Param(self, "k", "")
        self._setDefault(k=2)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, predictionCol=None, dictionary=None, k=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setDictionary(self, value):
        self._paramMap[self.dictionary] = value
        return self

    def getDictionary(self):
        return self.getOrDefault(self.dictionary)

    def setK(self, value):
        self._paramMap[self.k] = value
        return self

    def getK(self):
        return self.getOrDefault(self.k)
        
    def _transform(self, dataset):
        dictionary = self.getDictionary()
        # function to look up each word with its corresponding cluster
        def f(sentence):
            clusters = []
            for word in sentence:
                if word in dictionary:
                    clusters.append(dictionary[word])
            return clusters
        out_col = self.getPredictionCol()
        in_col = dataset[self.getInputCol()]
        t = ArrayType(StringType())
        transformed_dataset = dataset.withColumn(out_col, udf(f, t)(in_col))
        return transformed_dataset
    
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
    train.cache()
    
    ## DATA PROCESSING PIPELINE
    # Split at whitespace and characters that are not letter
    tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\P{Alpha}+")
    
    # stopword remover
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    pipeline_data_processing = Pipeline(stages=[tokenizer, remover])
    model_data_processing = pipeline_data_processing.fit(train)
    train_processed = model_data_processing.transform(train)
    train.unpersist()
    train_processed.cache()
    
    ## INTERMEDIATE STEP TO GET WORD VOCABULARY AND VECTOR
    # word2vec
    word2Vec = Word2Vec(inputCol="filtered_words", outputCol="word2vec_features")
    model_word2Vec = word2Vec.fit(train_processed)
    # Dataframe dictionary of Word-vectors
    vocabulary = model_word2Vec.getVectors()
    vocabulary.cache()
    
    ## ML PIPELINE
    # WordCluster Features
    wordcluster = WordCluster(inputCol="filtered_words", predictionCol="cluster", \
                              k=3, vocabulary=vocabulary)
    
    # get vector of cluster frequency for each document
    count_vectorizer = CountVectorizer(inputCol="cluster", outputCol="count")

    # normalized cluster frequency vector for each document
    normalizer = Normalizer(inputCol="count", outputCol="features", p=1.0)          
    
    # linear Regression Model
    lr = LinearRegression(maxIter=20, regParam=0.1)
    
    # Final Pipeline
    pipeline = Pipeline(stages=[wordcluster, count_vectorizer, normalizer, lr])
    
    ## FIT MODEL USING CROSS VALIDATION
    # Parameter grid for cross validation: numFeatures and regParam
    paramGrid = ParamGridBuilder() \
            .addGrid(wordcluster.k, [1000, 5000, 10000, 20000]) \
            .addGrid(lr.regParam, [0.001, 0.01, 0.1, 1.0]) \
            .build()
        
    # 5-fold cross validation
    evaluator = RegressionEvaluator(metricName="rmse")
    crossval = CrossValidator(estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=5)
    
    # Run cross-validation, and choose the best set of parameters.
    model = crossval.fit(train_processed)

    # RMSE on train data
    prediction_train = model.transform(train_processed)
    rmse_train = evaluator.evaluate(prediction_train)
    train_processed.unpersist()
    vocabulary.unpersist()
    
    ## TEST DATA
    #read test json file and process data (label, feature)
    text = sqlCt.read.json(inputs_test)
    test= text.select('overall','reviewText').withColumnRenamed('overall','label')
    test_processed = model_data_processing.transform(test)
    
    # Evaluate the model on test data
    prediction_test = model.transform(test_processed)
    rmse_test = evaluator.evaluate(prediction_test)
 
    # Print Result
    result = "MODEL WITH Word Clustering features - best k = " \
          + str(model.bestModel.stages[0].getK()) + ":\n"
    result = result + "-Train RMSE: " + str(rmse_train) + "\n"
    result = result + "-Test RMSE: " + str(rmse_test) + "\n"
    print(result)

if __name__ == "__main__":
    main()

    