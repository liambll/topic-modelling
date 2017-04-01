/* Author: linhb */

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.types.{StringType, ArrayType, DoubleType, IntegerType}

import scala.collection.JavaConverters._

object TopicDistribution {
  def main(args: Array[String]) {
    var output_path = "/user/llbui/bigdata"
    //var output_path = "C:/Users/linhb/bigdata"
    if (args.length > 0) {
      if (args(0).length > 0) {
        output_path = args(0)
      }  
    }
    
    val my_spark = SparkSession.builder()
      .master("local")
      .appName("Topic Model")
      //.config("spark.mongodb.input.uri", "mongodb://gateway.sfucloud.ca:27017/publications.papers")
      //.config("spark.mongodb.output.uri", "mongodb://gateway.sfucloud.ca:27017/publications.papers")
      .config("spark.mongodb.input.uri", "mongodb://localhost:27017/publications.papers")
      .config("spark.mongodb.output.uri", "mongodb://localhost:27017/publications.topicDistribution")
      .getOrCreate()
    my_spark.sparkContext.setLogLevel("WARN")
    
    // load topicDistribution
    val df_Document = my_spark.read.load(output_path + "/topicDistribution.parquet")
    val vectorHead = udf{ x:DenseVector => x.toArray }
    val df_Document2 = df_Document.withColumn("topic_distribution", vectorHead(df_Document("topicDistribution")))
    df_Document2.select("_id", "topic_distribution").write.format("com.mongodb.spark.sql.DefaultSource").mode("update").save()

  }
  
}