# CS643 - Programming Assignment 2: Wine Quality Prediction with Apache Spark and Docker on AWS

## Student Information
- **Name:** [Your Name]
- **Date:** [Submission Date]
- **Course:** CS643 - Programming Assignment 2

## Repository and Container Links
- **GitHub Repository:** [GitHub Link](https://github.com/jcardona321/CS643-WINEAPP)
- **Docker Hub Repository:** [Docker Hub Link](https://hub.docker.com/r/[YourDockerUsername]/[YourDockerRepository])

## Project Overview
This project involves the development of a parallel machine learning application for predicting wine quality using Apache Spark on AWS, and containerizing the application using Docker.

## Setup and Execution Instructions

### Step 1: AWS EMR Cluster Creation
- Create an EMR cluster with specified configurations.

### Step 2: Data and Application Upload
- Upload necessary datasets and python files to the master node of the cluster.

### Step 3: Model Training
- Execute the Spark job for model training on the EMR cluster.

### Step 4: Single EC2 Instance Prediction
- Set up and configure a single EC2 instance.
- Run the prediction application on this instance.

### Step 5: Docker-based Prediction
- Install Docker on the EC2 instance.
- Pull and run the Docker container for prediction.

## Grading Criteria
- Parallel Training Implementation: 50 points
- Single Machine Prediction Application: 25 points
- Docker Container for Prediction Application: 25 points
