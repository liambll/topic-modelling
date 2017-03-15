# -*- coding: utf-8 -*-

dictionary = model.stages[2].getDictionary()
len(dictionary)
import pandas as pd
df = pd.DataFrame(list(dictionary.items()),
                      columns=['word','cluster'])
cluster = dictionary['animal']
df_cluster = df[df['cluster']==cluster]
df_cluster

cluster = dictionary['good']
df_cluster = df[df['cluster']==cluster]
df_cluster
