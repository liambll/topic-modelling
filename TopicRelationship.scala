/* Author: linhb 
 * Some code logic on interestingness are from Advanced Analytics with Spark book
 */

// spark-submit --master yarn --deploy-mode client --driver-memory 10g --class TopicRelationship big-data-analytics-project_2.11-1.0.jar "" 0.0025 

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd._
import org.apache.spark.graphx._
import scala.collection.mutable.ListBuffer

object TopicRelationship {
  def main(args: Array[String]) {
    var input_path = "/user/llbui/bigdata45_500"
    //val output_path = "C:/Users/linhb/bigdata"
    var output_path = "/user/llbui/bigdatagraph"
    //val output_path = "C:/Users/linhb/bigdatagraph"
    
    var threshold = 0.0004795  // topic distribution threshold to decide if a document is considered belong to a topic
    var chisquare_threshold = 19.5  // Chisquare threshold to decide if two topics relationship is considered interesting
    var list_k: List[Int] = List()
    if (args.length > 0) {
      if (args(0).length > 0) {
        input_path = args(0)
      }  
    }
    if (args.length > 1) {
      threshold = args(1).toDouble
    }
    if (args.length > 2) {
      chisquare_threshold = args(2).toDouble
    }
    
    val my_spark = SparkSession.builder()
      //.master("local")
      .appName("Topic Relationship")
      .getOrCreate()
    my_spark.sparkContext.setLogLevel("WARN")
    
    // Read data into document - majorTopics RDD
    val rdd_Document: RDD[Array[Double]] = my_spark.read.load(input_path + "/topicDistribution.parquet").rdd
      .map(row => row(5).asInstanceOf[DenseVector])
      .map(_.toArray.map(_.toDouble))
    //println("- Topic Distribution stat:\n" +rdd_Document.flatMap(x => x).stats())
    // save topic distribution so that we can examine it and choose threshold at 80 percentile
    rdd_Document.flatMap(x => x).coalesce(1).saveAsTextFile(output_path + "/topicDistributionStat")
    
    val documentTopic: RDD[Seq[Int]] = rdd_Document.map(x => get_major_topics(threshold, x))
    documentTopic.cache()
    
    // Generate topics RDD and topic-pairs RDD
    val topics: RDD[Int] = documentTopic.flatMap(x => x)   
    val topicPairs = documentTopic.flatMap(t => t.sorted.combinations(2))
    val cooccurs = topicPairs.map(p => (p, 1)).reduceByKey(_+_)
    cooccurs.cache()
    
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
    topicGraph.edges.coalesce(1).saveAsTextFile(output_path + "/topicGraph")
    
    // Check degree
    val degrees: VertexRDD[Int] = topicGraph.degrees.cache()
    println("- Degree:\n" + degrees.map(_._2).stats())
    
    // Check connected components
    val connectedComponentGraph: Graph[VertexId, Int] = topicGraph.connectedComponents()
    val componentCounts = sortedConnectedComponents(connectedComponentGraph)
    println("- Connected Component")
    componentCounts.foreach(println)
    
    
    /* Graph based on Chi-square */
    // Create Graph
    val n = documentTopic.count()
    val topicCountsRdd = topics.map(x => (x.toLong, 1)).reduceByKey(_+_)
    val topicCountGraph = Graph(topicCountsRdd, topicGraph.edges)
    val chiSquaredGraph = topicCountGraph.mapTriplets(triplet => {
      chiSq(triplet.attr, triplet.srcAttr, triplet.dstAttr, n)
    })
    println("- Chisquare stat:\n" +chiSquaredGraph.edges.map(x => x.attr).stats())
    
    val interesting = chiSquaredGraph.subgraph(
      triplet => triplet.attr > chisquare_threshold)
    //interesting.edges.count
    interesting.edges.coalesce(1).saveAsTextFile(output_path + "/interestingGraph")
    
    // Check degree
    val interestingDegrees = interesting.degrees.cache()
    println("- Chisquare Degree:\n" +interestingDegrees.map(_._2).stats())
    
    // Check connected components
    val interestingComponentCounts = sortedConnectedComponents(interesting.connectedComponents())
    print(interestingComponentCounts.size)
    println("- Interesting Connected Component")
    interestingComponentCounts.foreach(println)
    
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
}