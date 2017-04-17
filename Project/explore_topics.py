# -*- coding: utf-8 -*-
# Read Topic-Words importance
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

my_spark = SparkSession \
    .builder \
    .appName("myApp") \
    .getOrCreate()
    
output_path = "C:/Users/linhb/bigdata45_500"
#output_path = "/user/llbui/bigdata40_1000"
topics_words = my_spark.read.parquet(output_path + "/topicwords.parquet")
topics_words.select("topic", "words").collect()

topics = topics_words.rdd.map(lambda row: (row[0], row[1], row[2], row[3])).collect()

# Visualize topics
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import pandas as pd
topic_lists = pd.read_csv('Project/topics.csv')

wc = WordCloud(width=200, height=200, background_color='white')
for i in range(len(topics)):
    topic = topics[i]
    frequencies = zip(topic[3], topic[2])
    wc.fit_words(frequencies=frequencies)
    plt.figure(i+1)
    plt.title("Topic: "+topic_lists.ix[topic[0],"Label"], loc='left')
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# Topic Concentration:
import matplotlib.pyplot as plt
import numpy as np
topic_concentration = np.array([36.83724211598519, 283.4710604189412, 53.357008114345696, 41.297608291495294, 52.57118366304121, 52.7470945499427, 279.2623495836199, 41.595591673414326, 221.3219784055919, 50.28797842351375, 79.03412887283523, 52.686018756646874, 189.39258458347206, 25.046094448617787, 71.18889861778415, 1219.036837257658, 39.07881904884451, 73.27518020450887, 681.9456687347974, 66.14776220901052, 53.19641556680988, 55.9700507888933, 50.56506364062264, 62.429442717817864, 288.2398840782747, 34.75379370558574, 67.39303535805398, 113.34341294007494, 27.792501323028908, 97.19695970049429, 77.9419143506714, 71.11989231727537, 92.2371638933898, 135.9778629069025, 168.46466404768023, 63.95978177131116, 1103.5521121694283, 47.18445900608829, 79.82968953842959, 80.13899139907397, 52.91302808120015, 59.5662045782038, 97.42700346282766, 86.24053326163394, 163.98505142215882])
topic_percentage = topic_concentration/6841*100
topic_names = topic_lists["Label"].values
order_index = np.argsort(topic_names)

fig, ax = plt.subplots(figsize=(5,15))
y_pos = np.arange(len(topic_names))
ax.barh(y_pos, topic_percentage[order_index], color='green', align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(topic_names[order_index])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('(%)')
ax.set_title('Overall Topic Contribution')

plt.show()
    
# Examine topic distribution
topics_dist = np.loadtxt("C:/Users/linhb/bigdatagraph/topicDistributionStat_45_500/part-00000", \
                   delimiter="\n", unpack=False)
np.percentile(topics_dist, 90)
min(topics_dist)
max(topics_dist)
np.median(topics_dist)
# the histogram of the data
n, bins, patches = plt.hist(topics_dist, 100, normed=1, facecolor='green', alpha=0.75)

plt.xlabel('Value (min=0.00001, max=0.996, median = 0.0001)')
plt.ylabel('Percentage')
plt.title(r'Topic Distribution Histogram')
plt.axis([0, 0.4, 0, 100])
plt.grid(True)

plt.show()


# Process topic relationship graph
import pandas as pd
topics_graph = pd.read_table("C:/Users/linhb/bigdatagraph/interestingGraph45_500/part-00000", \
                   sep=r'\(|,|\)', engine='python', header=None, names=["Edge","Source","Target","Weight","NA"])
topics_graph = topics_graph.loc[:,["Source","Target","Weight"]]
topics_graph["Type"] = "Undirected"
topics_graph.to_csv("C:/Users/linhb/bigdatagraph/interestingGraph45_500/graph.csv", \
                    index=False)


