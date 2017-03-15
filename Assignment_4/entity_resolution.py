# -*- coding: utf-8 -*-
# USAGE: spark-submit --master yarn-client --num-executors 10 entity_resolution.py
#    /user/llbui/a2/amazon-google 0.5 Amazon Google stopwords.txt Amazon_Google_perfectMapping
# First argument is a directory containing default "Amazon" and "Google" parquet files,
#   "stopwords.txt" and "Amazon_Google_perfectMapping" parquet file
# Second argument is threshold. Default is 0.5
# The last four arguments are only needed if one of the above file names are different

# entity_resolution.py
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, concat_ws
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql.types import FloatType
import sys

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        #self.f = open(stopWordsFile, "r")
        #self.stopWords = set(self.f.read().split("\n"))
        self.f = sqlCt.read.text(stopWordsFile)
        self.stopWords = self.f.map(lambda row: row.value).collect()
        self.stopWordsBC = sc.broadcast(self.stopWords)
        self.df1 = sqlCt.read.parquet(dataFile1).cache()
        self.df2 = sqlCt.read.parquet(dataFile2).cache()

    def preprocessDF(self, df, cols): 
        # concatenation
        df_concat = df.withColumn("concat", concat_ws(' ',*cols))
        
        # Split at whitespace and characters that are not letter
        tokenizer = RegexTokenizer(inputCol="concat", outputCol="words", pattern=r'\W+')
        df_tokenizer = tokenizer.transform(df_concat)
        
        # stopword remover
        remover = StopWordsRemover(inputCol="words", outputCol="joinKey", stopWords=self.stopWordsBC.value)
        df_remover = remover.transform(df_tokenizer) \
            .drop("concat").drop("words")
        return df_remover
        
    def filtering(self, df1, df2):
        # remove unnecessary fields
        df1_drop = df1.select("id", "joinKey").withColumnRenamed("joinKey", "joinKey1")
        df2_drop = df2.select("id", "joinKey").withColumnRenamed("joinKey", "joinKey2")
        
        # inverted index
        df_index1 = df1_drop.flatMap( lambda line: [(token, line['id']) for token in line['joinKey1']] ) \
            .toDF(["token", "id1"]).distinct()
        df_index2 = df2_drop.flatMap( lambda line: [(token, line['id']) for token in line['joinKey2']] ) \
            .toDF(["token", "id2"]).distinct()
        
        # merge to get id1 and id2 with common word
        df_cand_index = df_index1.join(df_index2, on="token").drop("token").distinct()
        
        # merge back joinKey1 and joinKey2
        df_cand = df_cand_index.join(df1_drop, df_cand_index.id1 == df1_drop.id).drop("id") \
            .join(df2_drop, df_cand_index.id2 == df2_drop.id).drop("id")
        return df_cand

    def verification(self, candDF, threshold):
        # Jaccard similarity function
        def jaccard_similarity(threshold, list1, list2):
            set1 = set(list1)
            set2 = set(list2)
            len1 = len(set1)
            len2 = len(set2)
            # skip computation of jaccard in extreme case or when min(set)/max(set) < threshold
            if ((min(len1,len2) == 0) or (1.0 * min(len1,len2)/max(len1,len2) < threshold)):
                return -1
            else:
                interset = set1.intersection(set2)
                return 1.0 * len(interset) / (len1 + len2 - len(interset))
        
        # get Jaccard index
        udf_jaccard = udf(lambda x,y: jaccard_similarity(threshold, x, y), FloatType())
        df_jaccard = candDF.withColumn("jaccard", udf_jaccard(candDF.joinKey1, candDF.joinKey2))
        return df_jaccard.filter(df_jaccard.jaccard >= threshold)
        
    def evaluate(self, result, groundTruth):
        # extreme case
        if ((len(result) == 0) or (len(groundTruth) == 0)):
            return (0, 0, 0)
        
        set_result = set(result)
        set_groundTruth = set(groundTruth)
        # True Positive
        TP = set_result.intersection(set_groundTruth)
        
        # Metrics
        precision = 1.0*len(TP)/len(set_result)
        recall = 1.0*len(TP)/len(set_groundTruth)
        fmeasure = 2.0*precision*recall/(precision+recall)
        return (precision, recall, fmeasure)

    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print("Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()))

        candDF = self.filtering(newDF1, newDF2)
        print("After Filtering: %d pairs left" %(candDF.count()))

        resultDF = self.verification(candDF, threshold)
        print("After Verification: %d similar pairs" %(resultDF.count()))

        return resultDF

    #def __del__(self):
        #self.f.close()


if __name__ == "__main__":
    # initialize Spark Context
    conf = SparkConf().setAppName('entity-resolution')
    sc = SparkContext(conf=conf)
    sqlCt = SQLContext(sc)
    
    # read input location
    inputs = sys.argv[1]
    # read threshold if provided
    if len(sys.argv) > 2:
        threshold = float(sys.argv[2])
    else:
        threshold = 0.5
    #load optional parameters if provided
    if len(sys.argv) > 6:
        dataFile1 = inputs + "/" + sys.argv[3]
        dataFile2 = inputs + "/" + sys.argv[4]
        stopWordsFile = inputs + "/" + sys.argv[5]
        perfectFile = inputs +  "/" + sys.argv[6]
    else:
        dataFile1 = inputs + "/Amazon"
        dataFile2 = inputs + "/Google"
        stopWordsFile = inputs + "/stopwords.txt"
        perfectFile = inputs + "/Amazon_Google_perfectMapping"
    
    # start Entity Resolution
    #er = EntityResolution("Amazon_sample", "Google_sample", "stopwords.txt")
    er = EntityResolution(dataFile1, dataFile2, stopWordsFile)
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, threshold)

    # Should we use dataframe instead of collect?
    result = resultDF.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = sqlCt.read.parquet(perfectFile) \
                          .map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))