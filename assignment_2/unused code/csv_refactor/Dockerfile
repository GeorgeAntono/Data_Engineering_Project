FROM openjdk:8

# Install Hadoop
RUN wget https://archive.apache.org/dist/hadoop/core/hadoop-2.10.1/hadoop-2.10.1.tar.gz && \
    tar -xzf hadoop-2.10.1.tar.gz -C /opt && \
    rm hadoop-2.10.1.tar.gz

# Set Hadoop environment variables
ENV HADOOP_HOME /opt/hadoop-2.10.1
ENV PATH $PATH:$HADOOP_HOME/bin

# Set the entry point
ENTRYPOINT ["/bin/bash"]
