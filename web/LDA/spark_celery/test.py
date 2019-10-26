# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:56:45 2017

@author: linhb
"""

from tasks import add, topicPredict
result = topicPredict.delay("gene deep learning")
output = result.get(timeout=60)
for paper in output:
    print(paper[1])

result.status



result = add.delay(4, 4)
result.get(timeout=5)