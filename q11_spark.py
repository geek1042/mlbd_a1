from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

spark = SparkSession.builder.appName("Q11_TFIDF").getOrCreate()

# Load 10 books
files = [
    "C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/200.txt",
    "C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/129.txt",
    "C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/30.txt",
    "C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/10.txt",
    "C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/180.txt",
    "C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/87.txt",
    "C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/48.txt",
    "C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/25.txt",
    "C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/115.txt",
    "C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/14.txt"
]

books_df = spark.read.text(files).withColumnRenamed("value", "text") \
    .withColumn("file_name", input_file_name())

# Tokenize text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(books_df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filteredData = remover.transform(wordsData)

# Term Frequency
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=500)
featurizedData = hashingTF.transform(filteredData)

# Inverse Document Frequency
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Show results
rescaledData.select("file_name", "features").show(5, truncate=False)

spark.stop()
