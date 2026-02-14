from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from pyspark.sql.functions import substring
from pyspark.sql.functions import regexp_extract
from pyspark.sql.functions import avg, length

# Create Spark session
spark = SparkSession.builder \
    .appName("Q10_Metadata") \
    .getOrCreate()

# Load dataset (whole file per row)
books_df = spark.read.format("text") \
    .option("wholetext", "true") \
    .load("/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/*.txt") \
    .withColumn("file_name", input_file_name())

books_df = books_df.withColumnRenamed("value", "text")

books_df = books_df.withColumn(
    "title",
    regexp_extract("text", r"Title:\s*([^\n\r]*)", 1)
)

books_df = books_df.withColumn(
    "release_date",
    regexp_extract("text", r"Release Date:.*?(\d{4})", 1)
)

books_df = books_df.withColumn(
    "language",
    regexp_extract("text", r"Language:\s*([^\n\r]*)", 1)
)

books_df = books_df.withColumn(
    "encoding",
    regexp_extract("text", r"Character set encoding:\s*([^\n\r]*)", 1)
)

print("Extracted Metadata:")
books_df.select("file_name", "title", "release_date", "language", "encoding").show(5, truncate=False)

print("Books per Year:")
books_df.groupBy("release_date").count().orderBy("release_date").show()

print("Most Common Language:")
books_df.groupBy("language").count().orderBy("count", ascending=False).show(1)

print("Average Title Length:")
books_df.select(avg(length("title"))).show()


spark.stop()
