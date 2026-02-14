from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, input_file_name
from pyspark.sql.functions import count


spark = SparkSession.builder \
    .appName("Q12_Author_Influence_Network") \
    .getOrCreate()

# 2. Load Dataset (whole file per row)
books_df = spark.read.format("text") \
    .option("wholetext", "true") \
    .load("/Users/nupurgupta/Documents/iit/ml_with_bigdata/assignment1/D184MB/*.txt") \
    .withColumn("file_name", input_file_name())

books_df = books_df.withColumnRenamed("value", "text")

# 3. Extract Author and Release Year
books_df = books_df.withColumn(
    "author",
    regexp_extract("text", r"Author:\s*([^\n\r]*)", 1)
)

books_df = books_df.withColumn(
    "release_year",
    regexp_extract("text", r"Release Date:.*?(\d{4})", 1)
)

# Keep only valid rows
books_df = books_df.filter(
    (col("author") != "") & (col("release_year") != "")
)

books_df = books_df.select("author", "release_year").dropDuplicates()

print("Extracted Author-Year Data:")
books_df.show(5, truncate=False)

# 4. Influence Network Construction
# Define time window
X = 5

a = books_df.alias("a")
b = books_df.alias("b")

edges = a.join(
    b,
    (col("a.author") != col("b.author")) &
    (col("b.release_year").cast("int") - col("a.release_year").cast("int") > 0) &
    (col("b.release_year").cast("int") - col("a.release_year").cast("int") <= X)
)

edges_df = edges.select(
    col("a.author").alias("author1"),
    col("b.author").alias("author2")
).dropDuplicates()

print("Influence Edges:")
edges_df.show(10, truncate=False)

# 5. Network Analysis

# In-degree (who was influenced)
in_degree = edges_df.groupBy("author2") \
    .agg(count("*").alias("in_degree"))

print("Top 5 Authors by In-Degree:")
in_degree.orderBy(col("in_degree").desc()).show(5, truncate=False)

# Out-degree (who influenced others)
out_degree = edges_df.groupBy("author1") \
    .agg(count("*").alias("out_degree"))

print("Top 5 Authors by Out-Degree:")
out_degree.orderBy(col("out_degree").desc()).show(5, truncate=False)

spark.stop()
