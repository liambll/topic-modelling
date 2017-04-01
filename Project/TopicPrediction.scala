/* Author: linhb */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, RegexTokenizer, StopWordsRemover, CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.functions.{udf,desc}
import org.apache.spark.sql.types.{StringType, ArrayType, DoubleType, IntegerType}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.clustering.{LDA, LocalLDAModel}
import scala.collection.mutable.{ListBuffer, ArrayBuffer}
import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import java.util.Properties
import scala.collection.JavaConverters._

// Start MongoDB Server: mongod.exe --dbpath D:\Training\Software\MongoDB\data
// /home/llbui/mongodb/mongodb-linux-x86_64-3.4.2/bin/mongod --dbpath /home/llbui/mongodb/data
// /home/llbui/mongodb/mongodb-linux-x86_64-3.4.2/bin/mongo mongodb://gateway.sfucloud.ca:27017
// module load spark/2.0.0
// Run job: spark-submit --master yarn --deploy-mode client --driver-memory 10g --executor-memory=10g --packages edu.stanford.nlp:stanford-corenlp:3.4.1,org.mongodb.spark:mongo-spark-connector_2.11:2.0.0 --jars stanford-corenlp-3.4.1-models.jar --class TopicModel big-data-analytics-project_2.11-1.0.jar

object TopicPrediction {
  def main(args: Array[String]) {
    //var output_path = "/user/llbui/bigdata"
    var output_path = "C:/Users/linhb/bigdata"
    var query = "gene analysis using deep learning" //query string
    var n = 10 // number of similar document to return
    val feature = "abstract" //feature to compare
    
    if (args.length > 0) {
      if (args(0).length > 0) {
        output_path = args(0)
      }  
    }    
    if (args.length > 1) {
      if (args(1).length > 0) {
        query = args(1)
      }  
    }
    if (args.length > 2) {
      n = args(2).toInt
    }  

    val my_spark = SparkSession.builder()
      .master("local")
      .appName("Topic Model")
      .getOrCreate()
    my_spark.sparkContext.setLogLevel("WARN")
    
    // clean up text
    val input_array = Array(query)
    val df = my_spark.createDataFrame(Seq((0, query))).toDF("id", feature)

    val udf_plainTextToLemmas = udf{s: String => plainTextToLemmas(s)}
    val df2 = df.withColumn("words", udf_plainTextToLemmas(df(feature)))
    
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("words2")
    val df3 = remover.transform(df2)
    
    
    val udf_remove_words = udf{s: Seq[String] => remove_words(s)}
    val df4 = df3.withColumn("words3", udf_remove_words(df3("words2")))   
    
    //text to feature vector - TF_IDF
    val countTF_model = CountVectorizerModel.load(output_path + "/tf_model")
    val df_countTF = countTF_model.transform(df4)
    
    val idf_model = IDFModel.load(output_path + "/idf_model")
    val df_IDF = idf_model.transform(df_countTF)
    df_IDF.cache()
    
    //LDA Model
    val lda_model = LocalLDAModel.load(output_path + "/lda_model")
    // output topics for document -> topicDistribution
    val df_Feature = lda_model.transform(df_IDF)
    val feature_vector = df_Feature.select("id", "topicDistribution").collect()(0)(1)
    println("Featuer Vector:" + feature_vector)
    
    //Load existing document
    val df_Document = my_spark.read.load(output_path + "/topicDistribution.parquet")
    val udf_cosineSimilarity = udf{x_vector: DenseVector => cosineSimilarity(x_vector, feature_vector.asInstanceOf[DenseVector])}
    val df_Similarity = df_Document.withColumn("similarity", udf_cosineSimilarity(df_Document("topicDistribution")))
    val df_Similarity_Sorted = df_Similarity.sort(desc("similarity"))
    //df_Similarity_Sorted.select("_id", "title", "similarity").collect.foreach(println)
    df_Similarity_Sorted.limit(n).select("_id", "title", "similarity").write.csv((output_path + "/documentSimilarity"))
    
  }
  
  def cosineSimilarity(x_vector: DenseVector, y_vector: DenseVector): Double = {
    val x = x_vector.toArray
    val y = y_vector.toArray
    require(x.size == y.size)
    val xy = (for((a, b) <- x zip y) yield a * b) sum
    val x2 = math.sqrt(x map(i => i*i) sum)
    val y2 = math.sqrt(x map(i => i*i) sum)
    if (x2*y2 == 0) {
      return 0
    }
    else
      return xy/(x2*y2)
  }
  
  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }
  
  def plainTextToLemmas(text: String): Seq[String] = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, pos, lemma")
    val pipeline = new StanfordCoreNLP(props)
    val doc = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences.asScala;
         token <- sentence.get(classOf[TokensAnnotation]).asScala) {
      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 && isOnlyLetters(lemma)) {
        lemmas += lemma.toLowerCase
      }
    }
    lemmas
  }
  
  def remove_words(aList: Seq[String]): Seq[String] = {
    val stopWords = List("abstract","keyword","introduction","conclusion","acknowledgement")
    return aList.filter(p => (p.length()>1 && !stopWords.contains(p)))
  }

  
  def lookup_words(termIndices: Seq[Integer], vocab:Array[String]): Seq[String] = {
    var words = ListBuffer[String]()
    for ( i <- termIndices) {
         words += vocab(i)
    }
    return words
  }
}