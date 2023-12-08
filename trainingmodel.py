import numpy as np
import pandas as pd
import warnings
import findspark
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize findspark and set configurations
findspark.init()
findspark.find()

conf = pyspark.SparkConf().setAppName('CS 643 Programming Assignment 2: Prediction').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

# Read and preprocess the dataset
df = spark.read.format("csv").load("TrainingDataset.csv", header=True, sep=";")
df.printSchema()
df.show(5)

for col_name in df.columns[1:-1] + ['""""quality"""""']:
    df = df.withColumn(col_name, col(col_name).cast('float'))
df = df.withColumnRenamed('""""quality"""""', "quality")

# Prepare features and labels
features = np.array(df.select(df.columns[1:-1]).collect())
label = np.array(df.select('quality').collect())

# Function to create labeled points
def to_labeled_point(features, labels):
    return [LabeledPoint(y, x) for x, y in zip(features, labels)]

# Function to convert to RDD
def to_rdd(sc, labeled_points):
    return sc.parallelize(labeled_points)

# Convert to labeled points and RDD
data_label_point = to_labeled_point(features, label)
data_label_point_rdd = to_rdd(sc, data_label_point)

# Split the data into training and testing sets
model_train, model_test = data_label_point_rdd.randomSplit([0.7, 0.3], seed=21)

# Train the Random Forest classifier
RF = RandomForest.trainClassifier(model_train, numClasses=10, categoricalFeaturesInfo={},
                                  numTrees=21, featureSubsetStrategy="auto",
                                  impurity='gini', maxDepth=30, maxBins=32)

# Make predictions
prediction = RF.predict(model_test.map(lambda x: x.features))
prediction_rdd = model_test.map(lambda y: y.label).zip(prediction)

# Convert predictions to DataFrame
quality_prediction = prediction_rdd.toDF(["quality", "prediction"])
quality_prediction.show()

# Convert to Pandas DataFrame for metrics calculation
quality_prediction_df = quality_prediction.toPandas()

# Output metrics
print("---------------Output-----------------")
accuracy = accuracy_score(quality_prediction_df['quality'], quality_prediction_df['prediction'])
f1 = f1_score(quality_prediction_df['quality'], quality_prediction_df['prediction'], average='weighted')
print(f"Accuracy : {accuracy}")
print(f"F1-score : {f1}")

test_error = prediction_rdd.filter(lambda y: y[0] != y[1]).count() / float(model_test.count())
print(f'Test Error : {test_error}')

# Save the model
RF.save(sc, 's3://pa2bucket/trainingmodel.model')
