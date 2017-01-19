# -*- coding: utf-8 -*-
# USAGE: spark-submit --master yarn-client matrix_multiply_sparse.py /user/llbui/a1-sparse /home/llbui
# The output output_sparse.txt will be written in the local output folder, eg. "/home/llbui"

from pyspark import SparkConf, SparkContext
from scipy.sparse import csr_matrix
import sys, operator, os

def create_sparse_matrix(vector):
   indptr = [0,len(vector)]
   indices = [int(i[0]) for i in vector]
   data = [float(i[1]) for i in vector]
   return csr_matrix((data,indices,indptr), shape=(1,100))
 
def main(argv=None):
    if argv is None:
        inputs = sys.argv[1]
        output = sys.argv[2]
    
    conf = SparkConf().setAppName('matrix-multiply-sparse')
    sc = SparkContext(conf=conf)
    
    #read input FILE
    text = sc.textFile(inputs)
    matrix_data = text.map(lambda line: line.split()) \
        .map(lambda vector:[i.split(':') for i in vector]) \
        .map(lambda vector:create_sparse_matrix(vector))
    
    # perform caculation using compressed representation  
    outer_product = matrix_data.map( lambda csr: csr.transpose().multiply(csr) )
    result = outer_product.reduce( operator.add )

    # Write Result to a local file
    if not os.path.exists(output):
        os.makedirs(output)
    f = open(output+'/output_sparse.txt', 'w')
    for i in range(0, result.get_shape()[0]):
        row = result.getrow(i)
        indices=row.indices
        data = row.data
        row_data = [str(item[0]) + ':' + str(item[1]) for item in zip(indices, data)]
        outdata = ' '.join(row_data)
        f.write("%s\n" % outdata)
    f.close()

if __name__ == "__main__":
    main()
    
    
    
        