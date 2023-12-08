import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import findspark
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import warnings
warnings.filterwarnings("ignore")

# Initialize Spark
findspark.init()
conf = pyspark.SparkConf().setAppName('CS 643 Programming Assignment 2: Prediction').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

# Load and preprocess training data
train_df = spark.read.format("csv").option("header", "true").option("sep", ";").load("TrainingDataset.csv")
for col_name in train_df.columns:
    train_df = train_df.withColumn(col_name.strip('"'), col(col_name).cast('float'))

# Prepare training data for RandomForest
train_features = np.array(train_df.select([col for col in train_df.columns if col != 'quality']).collect())
train_label = np.array(train_df.select('quality').collect()).flatten()
train_data_label_point = [LabeledPoint(y, x) for x, y in zip(train_features, train_label)]
train_data_rdd = sc.parallelize(train_data_label_point)

# Train RandomForest model
num_classes = 10
num_trees = 5
max_depth = 4
max_bins = 32
model = RandomForest.trainClassifier(train_data_rdd, numClasses=num_classes, categoricalFeaturesInfo={}, 
                                     numTrees=num_trees, featureSubsetStrategy="auto", 
                                     impurity='gini', maxDepth=max_depth, maxBins=max_bins)

# Save the trained model
model_save_path = "/home/hadoop/random_forest_model"
model.save(sc, model_save_path)

# Load and preprocess validation data
valid_df = spark.read.format("csv").option("header", "true").option("sep", ";").load("ValidationDataset.csv")
for col_name in valid_df.columns:
    valid_df = valid_df.withColumn(col_name.strip('"'), col(col_name).cast('float'))

# Prepare validation data
valid_features = np.array(valid_df.select([col for col in valid_df.columns if col != 'quality']).collect())
valid_label = np.array(valid_df.select('quality').collect()).flatten()
valid_data_label_point = [LabeledPoint(y, x) for x, y in zip(valid_features, valid_label)]
valid_data_rdd = sc.parallelize(valid_data_label_point)

# Load the model for prediction
RF = RandomForestModel.load(sc, model_save_path)

# Make predictions
predictions = RF.predict(valid_data_rdd.map(lambda x: x.features))
prediction_rdd = valid_data_rdd.map(lambda y: y.label).zip(predictions)

# Convert to DataFrame for evaluation
prediction_df = spark.createDataFrame(prediction_rdd, ["quality", "prediction"])

# Print the top 5 prediction results
print("Top 5 Prediction Results:")
prediction_df.show(5)

# Convert to Pandas DataFrame for further evaluation
quality_prediction_df = prediction_df.toPandas()

# Print results
print("---------------Results-----------------")
print("Accuracy : ", accuracy_score(quality_prediction_df['quality'], quality_prediction_df['prediction']))
print("F1-score : ", f1_score(quality_prediction_df['quality'], quality_prediction_df['prediction'], average='weighted'))
test_error = prediction_rdd.filter(lambda y: y[0] != y[1]).count() / float(valid_data_rdd.count())
print('Test Error : ' + str(test_error))
