# -*- coding: utf-8 -*-
# Read Topic-Words importance
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

my_spark = SparkSession \
    .builder \
    .appName("myApp") \
    .getOrCreate()
    
output_path = "C:/Users/linhb/bigdata"
#output_path = "/user/llbui/bigdata40_1000"
topics_words = my_spark.read.parquet(output_path + "/topicwords45_500")
topics_words.select("words").collect()

topics = topics_words.rdd.map(lambda row: (row[0], row[1], row[2], row[3])).collect()

# Visualize topics
from wordcloud import WordCloud
import matplotlib.pyplot as plt
topicName = []

wc = WordCloud(width=200, height=200, background_color='white')
for i in range(len(topics)):
    topic = topics[i]
    frequencies = zip(topic[3], topic[2])
    wc.fit_words(frequencies=frequencies)
    plt.figure(i+1)
    plt.title("Topic "+str(topic[0]), loc='left')
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# Topic Concentration:
topic_concentration = [0.016241542983799543,0.04282767692984204,0.017755791257006926,0.016490719451059344,0.01792654744916836,0.017785826015550236,0.05249897335286121,0.018516124330313008,0.037614901337096496,0.018281813061858745,0.02385836666306913,0.018996246933882906,0.026958235077199414,0.01328200624136438,0.023172434462727473,0.19803598173960907,0.018052923766033385,0.02026100693399446,0.10999860535236615,0.01937550265187473,0.021635495394392464,0.018118073261804264,0.018072645428638116,0.022095594544428013,0.03904118977315525,0.01533320740783718,0.02331014567921863,0.02229587299449891,0.014618915994930192,0.025095381875353526,0.02404469723848798,0.020804216751882435,0.024311300491314355,0.032666021051562195,0.029641009603462016,0.024015737438795995,0.19867163464786908,0.016733468555542294,0.029075078215526594,0.022464854824805962,0.021671677646674537,0.019483150184523505,0.029429833134220602,0.028691285844165644,0.030758752920157673]
    
    
# Examine topic distribution
import numpy as np
topics_dist = np.loadtxt("C:/Users/linhb/bigdatagraph/topicDistributionStat_45_500/part-00000", \
                   delimiter="\n", unpack=False)
np.percentile(topics_dist, 90)


# Process topic relationship graph
import pandas as pd
topics_graph = pd.read_table("C:/Users/linhb/bigdatagraph/interestingGraph45_500/part-00000", \
                   sep=r'\(|,|\)', engine='python', header=None, names=["Edge","Source","Target","Weight","NA"])
topics_graph = topics_graph.loc[:,["Source","Target","Weight"]]
topics_graph["Type"] = "Undirected"
topics_graph.to_csv("C:/Users/linhb/bigdatagraph/interestingGraph45_500/graph.csv", \
                    index=False)


