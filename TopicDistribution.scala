/* Author: linhb */

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.sum
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.types.{StringType, ArrayType, DoubleType, IntegerType}

import scala.collection.JavaConverters._

object TopicDistribution {
  def main(args: Array[String]) {
    var output_path = "/user/llbui/bigdata45_500"
    //var output_path = "C:/Users/linhb/bigdata"
    if (args.length > 0) {
      if (args(0).length > 0) {
        output_path = args(0)
      }  
    }
    
    val my_spark = SparkSession.builder()
      //.master("local")
      .appName("Topic Distribution")
      //.config("spark.mongodb.input.uri", "mongodb://gateway.sfucloud.ca:27017/publications.papers")
      .config("spark.mongodb.output.uri", "mongodb://gateway.sfucloud.ca:27017/publications.topicDistribution")
      //.config("spark.mongodb.input.uri", "mongodb://localhost:27017/publications.papers")
      //.config("spark.mongodb.output.uri", "mongodb://localhost:27017/publications.topicDistribution")
      .getOrCreate()
    my_spark.sparkContext.setLogLevel("WARN")
    
    // load topicDistribution
    val df_Document = my_spark.read.load(output_path + "/topicDistribution.parquet")
    val vectorHead = udf{ x:DenseVector => x.toArray }
    val df_Document2 = df_Document.withColumn("topic_distribution", vectorHead(df_Document("topicDistribution")))
    
    // overall topic distribution of over whole corpus
    //val rdd_Document = df_Document2.select("topic_distribution").rdd.map(row => row(0).asInstanceOf[Seq[Double]]).reduce(sum_vectors)
    //print(rdd_Document)
    
    //Save topicDistribution to MongoDB
    df_Document2.drop("topicDistribution").write.format("com.mongodb.spark.sql.DefaultSource").mode("append").save()

  }
  
  def sum_vectors(x: Seq[Double], y: Seq[Double]): Seq[Double] = {
    return (x, y).zipped.map(_+_)
  }
  
}