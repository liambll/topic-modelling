# -*- coding: utf-8 -*-
# USAGE: spark-submit --master yarn-client matrix_multiply.py /user/llbui/a1 /home/llbui
# The output output.txt will be written in the local output folder, eg."/home/llbui"

from pyspark import SparkConf, SparkContext
import numpy as np
import sys, operator, os

def main(argv=None):
    if argv is None:
        inputs = sys.argv[1]
        output = sys.argv[2]

    conf = SparkConf().setAppName('matrix-multiply')
    sc = SparkContext(conf=conf)
    
    #read input FILE
    text = sc.textFile(inputs)
    matrix_data = text.map(lambda line: line.split()) \
        .map(lambda vector:[float(i) for i in vector]) 
    
    # perform caculation   
    outer_product = matrix_data.map(lambda vector: np.outer(vector,vector))
    result = outer_product.reduce(operator.add)

    # Write Result to a local file
    if not os.path.exists(output):
        os.makedirs(output)
    np.savetxt(output+'/output.txt', result, fmt="%s")

if __name__ == "__main__":
    main()
    
        