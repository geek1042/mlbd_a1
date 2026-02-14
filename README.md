# Machine Learning with Big Data -- Assignment 1

**Course Code:** CSL7110\
**Topic:** Hadoop MapReduce and Apache Spark Big Data Processing

------------------------------------------------------------------------

## 1. Overview

This repository contains the code and documentation for Assignment 1 of
the Machine Learning with Big Data course. The assignment covers Hadoop
MapReduce programming, HDFS concepts, Apache Spark text analytics,
metadata extraction, TF-IDF computation, and author influence network
analysis using the Project Gutenberg dataset.

The objective is to understand distributed data processing, parallel
computation, and large-scale text analytics using Hadoop and Spark
frameworks.

------------------------------------------------------------------------

## 2. System Requirements

### Software Requirements

-   Ubuntu Linux (via VirtualBox)\
-   Java 8 or Java 11\
-   Apache Hadoop 3.x (Single Node Cluster)\
-   Apache Spark\
-   Python 3.x with PySpark\
-   Git

### Dataset

-   Project Gutenberg dataset (D184MB) containing multiple `.txt` book
    files.\
-   Dataset is excluded from GitHub due to size limitations.

------------------------------------------------------------------------

## 3. Hadoop Setup Summary

Steps followed to configure Hadoop Single Node Cluster:

1.  Install Java and configure JAVA_HOME.\
2.  Download and extract Hadoop.\
3.  Configure environment variables: HADOOP_HOME, PATH.\
4.  Edit `core-site.xml` and `hdfs-site.xml`.\
5.  Enable SSH localhost access.\
6.  Format HDFS using `hdfs namenode -format`.\
7.  Start Hadoop using `start-dfs.sh`.\
8.  Verify running services using `jps`.

------------------------------------------------------------------------

## 4. MapReduce WordCount Execution

### Steps

1.  Copy input file to HDFS:

    ``` bash
    hdfs dfs -copyFromLocal input.txt /user/input/
    ```

2.  Compile and run WordCount Java program.\

3.  Execute job:

    ``` bash
    hadoop jar WordCount.jar /user/input /user/output
    ```

4.  Retrieve output:

    ``` bash
    hdfs dfs -getmerge /user/output output.txt
    ```

5.  Verify word frequency results.

------------------------------------------------------------------------

## 5. HDFS Replication and Input Split Analysis

-   HDFS replicates files (default replication factor = 3) for fault
    tolerance.\
-   Directories are not replicated because they store metadata, not data
    blocks.\
-   Input split size affects the number of mappers and execution time.\
    Smaller split size increases parallelism but also overhead.\
    Larger split size reduces mapper count but may reduce parallelism.

------------------------------------------------------------------------

## 6. Metadata Extraction Using Apache Spark

### Method

1.  Load each book as a single row using `wholetext=true`.\
2.  Extract metadata using regular expressions:
    -   Title\
    -   Release Date\
    -   Language\
    -   Encoding\
3.  Store extracted metadata in Spark DataFrame.

### Analysis Performed

-   Number of books released per year\
-   Most common language\
-   Average title length

------------------------------------------------------------------------

## 7. TF-IDF and Book Similarity

### Processing Steps

1.  Remove Project Gutenberg headers and footers.\
2.  Convert text to lowercase and remove punctuation.\
3.  Tokenize text and remove stopwords.\
4.  Compute Term Frequency (TF).\
5.  Compute Inverse Document Frequency (IDF).\
6.  Calculate TF-IDF vectors for each book.\
7.  Compute cosine similarity to find similar books.

TF-IDF is used to weight important words in documents, and cosine
similarity is used to measure document similarity in vector space.

------------------------------------------------------------------------

## 8. Author Influence Network

### Method

1.  Extract author name and release year using regex.\
2.  Define influence if two authors published within X years.\
3.  Construct directed graph edges (author1 → author2).\
4.  Compute in-degree and out-degree for authors.

This is a simplified influence model used for graph-based analysis
demonstration.

------------------------------------------------------------------------

## 9. Repository Structure

    mlbd_a1/
    │
    ├── code/                # Java and PySpark source code
    ├── report/              # Assignment PDF and documentation
    ├── results/             # Output files and screenshots
    ├── README.md
    └── .gitignore           # Dataset excluded

------------------------------------------------------------------------

## 10. How to Run

### Hadoop WordCount

``` bash
hadoop jar WordCount.jar input output
```

### Spark Metadata Extraction

``` bash
spark-submit q10_metadata.py
```

### Spark TF-IDF

``` bash
spark-submit q11_tfidf.py
```

### Author Influence Network

``` bash
spark-submit q12_author_network.py
```

------------------------------------------------------------------------

## 11. Notes and Limitations

-   Regex-based metadata extraction may fail due to inconsistent file
    formatting.\
-   Some books have missing or noisy metadata fields.\
-   Dataset is excluded from GitHub due to file size constraints (\>100
    MB).\
-   Advanced NLP methods can improve metadata extraction in real-world
    systems.

------------------------------------------------------------------------

## 12. Author

**Name:** Shreyas B. Gaikwad\
**Roll Number:** M25DE1042\
**Course:** Machine Learning with Big Data (CSL7110)

------------------------------------------------------------------------
