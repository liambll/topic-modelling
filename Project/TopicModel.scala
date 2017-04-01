/* Author: linhb */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, CountVectorizer}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StringType, ArrayType, DoubleType, IntegerType}
import org.apache.spark.ml.clustering.{LDA, LDAModel}
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

object TopicModel {
  def main(args: Array[String]) {
    var output_path = "/user/llbui/bigdata"
    //var output_path = "C:/Users/linhb/bigdata"
    var feature = "abstract" //feature to perform LDA on
    var mode = "online" //online or em LDA mode
    var maxIter = 100
    var initial_k = 2 //number of topics
    var list_k: List[Int] = List() //other number of topics to try
    if (args.length > 0) {
      if (args(0).length > 0) {
        output_path = args(0)
      }  
    }
    if (args.length > 1) {
      if (args(1) == "text") {
        feature = "text"
      }  
    }
    if (args.length > 2) {
      if (args(2) == "em") {
        mode = "em"
      }  
    }
    if (args.length > 3) {
      maxIter = args(3).toInt
    }    
    if (args.length > 4) {
      initial_k = args(4).toInt
    }
    if (args.length > 5) {
      list_k = args.slice(5, args.length).toList.map { x => x.toInt }
    }

    val my_spark = SparkSession.builder()
      //.master("local")
      .appName("Topic Model")
      .config("spark.mongodb.input.uri", "mongodb://gateway.sfucloud.ca:27017/publications.papers")
      .config("spark.mongodb.output.uri", "mongodb://gateway.sfucloud.ca:27017/publications.papers")
      //.config("spark.mongodb.input.uri", "mongodb://localhost:27017/publications.papers")
      //.config("spark.mongodb.output.uri", "mongodb://localhost:27017/publications.papers")
      .getOrCreate()
    my_spark.sparkContext.setLogLevel("WARN")
    
    var output_log = ""
    
    // clean up text
    val df_mongo = my_spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
    val df = df_mongo.drop("content")

    val udf_plainTextToLemmas = udf{s: String => plainTextToLemmas(s)}
    val df2 = df.withColumn("words", udf_plainTextToLemmas(df(feature)))
    
    /*
    val tokenizer = new RegexTokenizer()
      .setInputCol("text_main")
      .setOutputCol("words")
      .setPattern("\\P{Alpha}+")
    val df2 = tokenizer.transform(df1)
    * 
    */
    
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("words2")
    val df3 = remover.transform(df2)
    
    
    val udf_remove_words = udf{s: Seq[String] => remove_words(s)}
    val df4 = df3.withColumn("words3", udf_remove_words(df3("words2")))   
    
    //text to feature vector - TF_IDF
    //should not use HashingTF because we need word-index dictionary
    val countTF = new CountVectorizer()
      .setInputCol("words3")
      .setOutputCol("raw_features")
      .setMinTF(1.0)
      .setMinDF(1.0)
    val countTF_model = countTF.fit(df4)
    countTF_model.save(output_path + "/tf_model")
    val df_countTF = countTF_model.transform(df4)
    val vocab = countTF_model.vocabulary
    output_log += "CountVectorizer vocab size: " + vocab.length + "\n"
    
    val idf = new IDF()
      .setInputCol("raw_features")
      .setOutputCol("features")
    val idf_model = idf.fit(df_countTF)
    idf_model.save(output_path + "/idf_model")
    val df_IDF = idf_model.transform(df_countTF)
    df_IDF.cache()
    
    //LDA Model
    var best_k = initial_k
    val lda = new LDA()
        .setK(best_k)
        .setSeed(1)
        .setOptimizer(mode)
        .setMaxIter(maxIter)
        .setFeaturesCol("features")
    var best_model = lda.fit(df_IDF)

    //evaluation: high likelihood, low perplexity
    val loglikelihood = best_model.logLikelihood(df_IDF)
    var best_LogLikelihood = loglikelihood
    output_log += "- k = " + best_k + "\n"
    output_log += "  LogPerplexity: " + best_model.logPerplexity(df_IDF) + "\n"
    output_log += "  LogLikelihood: " + loglikelihood + "\n"
    
    for (k <- list_k) {
      val lda = new LDA()
        .setK(k)
        .setSeed(1)
        .setOptimizer(mode)
        .setMaxIter(maxIter)
        .setFeaturesCol("features")
      val lda_model = lda.fit(df_IDF)

      //evaluation: high likelihood, low perplexity
      val loglikelihood = lda_model.logLikelihood(df_IDF)
      output_log += "- k = " + k + "\n"
      output_log += "  LogPerplexity: " + lda_model.logPerplexity(df_IDF) + "\n"
      output_log += "  LogLikelihood: " + loglikelihood + "\n"
      
      if (loglikelihood > best_LogLikelihood) {
        best_model = lda_model
        best_k = k
        best_LogLikelihood = loglikelihood
      }
    }
    //save best model
    output_log += "Best k = " + best_k + "\n"
    output_log += "Best LogPerplexity: " + best_model.logPerplexity(df_IDF) + "\n"
    output_log += "Best LogLikelihood: " + best_LogLikelihood + "\n"
    best_model.save(output_path + "/lda_model")
    
    //LDA model description
    println("LDA Model vocab size:" + best_model.vocabSize)
    val topics = best_model.describeTopics(maxTermsPerTopic=30)
    //topics.show()
    val udf_lookup_words = udf{x:Seq[Integer]  => lookup_words(x, vocab)}
    val topics_words = topics.withColumn("words", udf_lookup_words(topics("termIndices")))
    //topics_words.show()
    topics_words.write.save(output_path + "/topicWords.parquet")
    
    //println(best_model.topicsMatrix)
    output_log += "Document Concentration: " + best_model.estimatedDocConcentration + "\n"
    
    // output topics for document -> topicDistribution
    val df_Document = best_model.transform(df_IDF)
    //println(df_Document.select("topicDistribution").head())
    // save topicDistribution for each document
    df_Document.select("_id", "authors", "title", "abstract", "url", "topicDistribution").write.save(output_path + "/topicDistribution.parquet")
    // save topicDistribution to mongoDB?
    output_log += "Done"
    
    //scala.tools.nsc.io.File("output_log.txt").writeAll(output_log)
    val output_array = Array(output_log)
    val output = my_spark.sparkContext.parallelize(output_array)
    output.coalesce(1).saveAsTextFile(output_path + "/output")
    
    println(output_log)
    
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