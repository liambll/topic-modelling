import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, CountVectorizer}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StringType, ArrayType, DoubleType, IntegerType}
import org.apache.spark.ml.clustering.LDA
import scala.collection.mutable.ListBuffer

//Start MongoDB Server: mongod.exe --dbpath D:\Training\Software\MongoDB\data

object MLPipeline {
  def main(args: Array[String]) {

    val my_spark = SparkSession.builder()
      .master("local")
      .appName("Topic Model")
      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/publications.papers")
      .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/publications.papers")
      .getOrCreate()
    my_spark.sparkContext.setLogLevel("WARN")
      
    // clean up text
    val df_mongo = my_spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
    val df = df_mongo.drop("content")
    
    val udf_extract_content = udf{s: String => extract_main_content(s)}
    val df1 = df.withColumn("text_main", udf_extract_content(df("text")))

    val tokenizer = new RegexTokenizer()
      .setInputCol("text_main")
      .setOutputCol("words")
      .setPattern("\\P{Alpha}+")
    val df2 = tokenizer.transform(df1)
    
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
    val df_countTF = countTF_model.transform(df4)
    val vocab = countTF_model.vocabulary
    println("CountVectorizer vocab size: " + vocab.length)
    
    val idf = new IDF()
      .setInputCol("raw_features")
      .setOutputCol("features")
    val idf_model = idf.fit(df_countTF)
    val df_IDF = idf_model.transform(df_countTF)
    df_IDF.cache()
    
    
    //LDA Model
    val lda = new LDA()
      .setK(2)
      .setSeed(1)
      .setOptimizer("online")
      .setMaxIter(100)
      .setFeaturesCol("features")
    val lda_model = lda.fit(df_IDF)

    //evaluation: high likelihood, low perplexity
    println("LogPerplexity: " + lda_model.logPerplexity(df_IDF))
    println("LogLikelihood: " + lda_model.logLikelihood(df_IDF))
    
    //LDA model description
    println("LDA Model vocab size:" + lda_model.vocabSize)
    val topics = lda_model.describeTopics(maxTermsPerTopic=10)
    //topics.show()
    val udf_lookup_words = udf{x:Seq[Integer]  => lookup_words(x, vocab)}
    val topics_words = topics.withColumn("words", udf_lookup_words(topics("termIndices")))
    topics_words.show()
    
    //println(lda_model.topicsMatrix)
    //println(lda_model.estimatedDocConcentration)
    
    // output topics for document -> topicDistribution
    //val df_Document = lda_model.transform(df_IDF)
    //println(df_Document.select("topicDistribution").head())
    
  }
  
  def extract_main_content(text: String): String = {
    //Main Content are between "Abstract" and "References"
    val text_lower = text.toLowerCase()
    var abstract_index = text_lower.indexOf("abstract")
    if (abstract_index == -1) {
        abstract_index = 0
    }
    var reference_index = text_lower.lastIndexOf("references")
    if (reference_index == -1) {
        reference_index = text_lower.length()
    }
    return text_lower.substring(abstract_index,reference_index)
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