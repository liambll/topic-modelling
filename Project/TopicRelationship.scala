/* Author: linhb 
 * Some code logic on interestingness are from Advanced Analytics with Spark book
 */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd._
import org.apache.spark.graphx._
import scala.collection.mutable.ListBuffer

object TopicRelationship {
  def main(args: Array[String]) {
    val output_path = "C:/Users/linhb/bigdata"

    val my_spark = SparkSession.builder()
      .master("local")
      .appName("Topic Relationship")
      //.config("spark.mongodb.input.uri", "mongodb://127.0.0.1/publications.papers")
      //.config("spark.mongodb.output.uri", "mongodb://127.0.0.1/publications.papers")
      .getOrCreate()
    my_spark.sparkContext.setLogLevel("WARN")
    
    // Read data into document - majorTopics RDD
    val rdd_Document: RDD[Array[Double]] = my_spark.read.load(output_path + "/topicDistribution.parquet").rdd
      .map(row => row(1).asInstanceOf[DenseVector])
      .map(_.toArray.map(_.toDouble))  
    
    //for (i <- 0 to 7) { print(rdd_Document.collect()(i).asInstanceOf[Array[Double]].mkString(" ")) }
    
    val threshold = 0.000024
    //val threshold = 0.0
    val documentTopic: RDD[Seq[Int]] = rdd_Document.map(x => get_major_topics(threshold, x))
    documentTopic.cache()
    //for (i <- 0 to 7) { print(documentTopic.collect()(i)) }
    
    // Generate topics RDD and topic-pairs RDD
    val topics: RDD[Int] = documentTopic.flatMap(x => x)   
    val topicPairs = documentTopic.flatMap(t => t.sorted.combinations(2))
    val cooccurs = topicPairs.map(p => (p, 1)).reduceByKey(_+_)
    cooccurs.cache()
    //print(cooccurs.collect()(0))
    
    /* Graph based on co-occurence */
    // Create Graph
    val vertices = topics.map(topic => (topic.toLong, topic))
    val edges = cooccurs.map(p => {
      val (topics, cnt) = p
      val ids = topics.map(topic => topic.toLong).sorted
      Edge(ids(0), ids(1), cnt)
    })
    val topicGraph = Graph(vertices, edges)
    topicGraph.cache()
    
    // Check degree
    val degrees: VertexRDD[Int] = topicGraph.degrees.cache()
    //print(degrees.map(_._2).stats())
    
    // Check connected components
    val connectedComponentGraph: Graph[VertexId, Int] = topicGraph.connectedComponents()
    val componentCounts = sortedConnectedComponents(connectedComponentGraph)
    //componentCounts.take(10)foreach(println)
    
    /* Graph based on Chi-square */
    // Create Graph
    val n = documentTopic.count()
    val topicCountsRdd = topics.map(x => (x.hashCode().toLong, 1)).reduceByKey(_+_)
    val topicCountGraph = Graph(topicCountsRdd, topicGraph.edges)
    val chiSquaredGraph = topicCountGraph.mapTriplets(triplet => {
      chiSq(triplet.attr, triplet.srcAttr, triplet.dstAttr, n)
    })
    print(chiSquaredGraph.edges.map(x => x.attr).stats())
    
    val interesting = chiSquaredGraph.subgraph(
      triplet => triplet.attr > 19.5)
    interesting.edges.count
    
    // Check degree
    val interestingDegrees = interesting.degrees.cache()
    print(interestingDegrees.map(_._2).stats())
    
    // Check connected components
    val interestingComponentCounts = sortedConnectedComponents(interesting.connectedComponents())
    print(interestingComponentCounts.size)
    interestingComponentCounts.take(10).foreach(println)
    
    // Clustering
    val avgCC = avgClusteringCoef(interesting)
    print(avgCC)
  }
  
  def get_major_topics(threshold: Double, topicDist: Array[Double]) : Seq[Int] = {
    var major_topics= ListBuffer[Int]()
    for ( i <- 0 to topicDist.length-1) {
         if (topicDist(i) >= threshold) {
           major_topics += i
         }
    }
    return major_topics.toList
  }
  
  def sortedConnectedComponents(connectedComponents: Graph[VertexId, _]): Seq[(VertexId, Long)] = {
      val componentCounts = connectedComponents.vertices.map(_._2).countByValue
      componentCounts.toSeq.sortBy(_._2).reverse
  }
  
  def chiSq(YY: Int, YB: Int, YA: Int, T: Long): Double = {
      val NB = T - YB
      val NA = T - YA
      val YN = YA - YY
      val NY = YB - YY
      val NN = T - NY - YN - YY
      val inner = (YY * NN - YN * NY) - T / 2.0
      T * math.pow(inner, 2) / (YA * NA * YB * NB)
  }
  
  def avgClusteringCoef(graph: Graph[_, _]): Double = {
    val triCountGraph = graph.triangleCount()
    val maxTrisGraph = graph.degrees.mapValues(d => d * (d - 1) / 2.0)
    val clusterCoefGraph = triCountGraph.vertices.innerJoin(maxTrisGraph) {
      (vertexId, triCount, maxTris) => if (maxTris == 0) 0 else triCount / maxTris
    }
    clusterCoefGraph.map(_._2).sum() / graph.vertices.count()
  } 
}