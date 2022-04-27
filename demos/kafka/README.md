# kafka-light-bulb

This repository provides code for [this](https://dev.to/boyu1997/intro-to-kafka-4hn2) simple Kafka tutorial.

## Setup

To execute the code, you will need:

- **Git**, install [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- **Docker**, install [here](https://docs.docker.com/get-docker/)
- **Docker Compose**, install [here](https://docs.docker.com/compose/install/)

Clone this repository
```
git clone git@github.com:Boyu1997/kafka-light-bulb.git
cd kafka-light-bulb
```

Start the Kafka cluster
```
docker-compose up
```
*Note: if you encountered issue and need to restart the Kafka cluster, make sure to delete the data folder by running `rm -rf data` before restarting using `docker-compose up`.*

Setup virtual environment for running Python code
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Run the producer script, the producer takes three optional inputs
- `--key="KEY_STRING"`, Kafka log key name, default to `1`
- `--topic="TOPIC_STRING"`, Kafka topic to publish to, default to `light_bulb`
- `--string="DATA_STRING"`, string data to be sended as Morse code, the string data must contain only character A-Z, default to `ABC`
```
python3 producer.py
python3 producer.py --key="123" `--STRING="XYZ"`
```

Run the consumer script
```
python3 consumer.py
```