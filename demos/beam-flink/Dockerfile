FROM python:3.8

WORKDIR /code
COPY ./* /code/

# Install OpenJDK-11
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME

RUN pip install -r /code/requirements.txt

CMD ["ls -alh"]