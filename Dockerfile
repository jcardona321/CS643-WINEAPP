FROM ubuntu:20.04

# Install Java, Python, and other dependencies
RUN apt-get update -y \
    && apt-get install openjdk-11-jdk -y \
    && apt-get install python3-pip -y \
    && apt-get install -y wget

# Download and install Spark
RUN wget https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz \
    && tar xvf spark-3.5.0-bin-hadoop3.tgz -C /opt \
    && rm spark-3.5.0-bin-hadoop3.tgz

# Set up environment variables
ENV SPARK_HOME=/opt/spark-3.5.0-bin-hadoop3
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=python3

# Create a non-root user
RUN useradd -ms /bin/bash myuser1
WORKDIR /home/myuser1
USER myuser1

# Copy your application files
COPY --chown=myuser1:myuser1 . /home/myuser1

# Install Python dependencies
RUN python3 -m pip install --user -r requirements.txt

# Set the entry point and command
ENTRYPOINT ["python3"]
CMD ["model_prediction.py"]
