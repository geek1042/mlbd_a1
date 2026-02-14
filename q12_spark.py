from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, input_file_name, count

# Create Spark session
spark = SparkSession.builder.appName("Q12_Author_Influence_Network").getOrCreate()

# Load dataset (whole book per row)
books_df = spark.read.format("text") \
    .option("wholetext", "true") \
    .load("C:/Users/Shreyas/OneDrive/Desktop/mlbd_a1/D184MB/*.txt") \
    .withColumn("file_name", input_file_name()) \
    .withColumnRenamed("value", "text")

# Extract Author
books_df = books_df.withColumn(
    "author",
    regexp_extract("text", r"Author:\s*([^\n\r]*)", 1)
)

# Extract Release Year
books_df = books_df.withColumn(
    "release_year",
    regexp_extract("text", r"Release Date:.*?(\d{4})", 1)
)

# Filter valid records
books_df = books_df.filter(
    (col("author") != "") & (col("release_year") != "")
)

# Unique author-year pairs
books_df = books_df.select("author", "release_year").dropDuplicates()

print("Extracted Author-Year Data:")
books_df.show(5, truncate=False)

# Define influence window
X = 5

# Self join for influence edges
a = books_df.alias("a")
b = books_df.alias("b")

edges = a.join(
    b,
    (col("a.author") != col("b.author")) &
    ((col("b.release_year").cast("int") - col("a.release_year").cast("int")) > 0) &
    ((col("b.release_year").cast("int") - col("a.release_year").cast("int")) <= X)
)

edges_df = edges.select(
    col("a.author").alias("author1"),
    col("b.author").alias("author2")
).dropDuplicates()

print("Influence Edges:")
edges_df.show(10, truncate=False)

# In-degree
in_degree = edges_df.groupBy("author2").agg(count("*").alias("in_degree"))
print("Top 5 Authors by In-Degree:")
in_degree.orderBy(col("in_degree").desc()).show(5, truncate=False)

# Out-degree
out_degree = edges_df.groupBy("author1").agg(count("*").alias("out_degree"))
print("Top 5 Authors by Out-Degree:")
out_degree.orderBy(col("out_degree").desc()).show(5, truncate=False)

spark.stop()
