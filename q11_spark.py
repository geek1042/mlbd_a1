from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF

spark = SparkSession.builder \
    .appName("Q11_TFIDF") \
    .getOrCreate()

# Load only 10 files
books_df = spark.read.text([
    "/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/200.txt",
    "/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/129.txt",
    "/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/30.txt",
    "/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/10.txt",
    "/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/180.txt",
    "/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/87.txt",
    "/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/48.txt",
    "/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/25.txt",
    "/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/115.txt",
    "/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/14.txt"
]).withColumnRenamed("value", "text") \
 .withColumn("file_name", input_file_name())

books_df = books_df.withColumnRenamed("value", "text")

# Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(books_df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filteredData = remover.transform(wordsData)

# TF
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=500)
featurizedData = hashingTF.transform(filteredData)

# IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

print("TF-IDF Features:")
rescaledData.select("file_name", "features").show(5, truncate=False)

spark.stop()
