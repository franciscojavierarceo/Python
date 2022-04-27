import json

from confluent_kafka import Consumer, KafkaError

c = Consumer({
    'bootstrap.servers': 'localhost:9091',
    'group.id': 'counting-group',
    'client.id': 'client-1',
    'enable.auto.commit': True,
    'session.timeout.ms': 6000,
    'default.topic.config': {'auto.offset.reset': 'smallest'}
})

c.subscribe(['light_bulb'])

last_timestamp = -1
count = {
    'short': 0,
    'long': 0
}

try:
    while True:
        msg = c.poll(0.1)
        if msg is None:
            continue
        elif not msg.error():
            data = json.loads(msg.value())
            if data['new_status'] == 'on':
                last_timestamp = data['timestamp']
            elif last_timestamp != -1:
                if data['timestamp'] - last_timestamp > 0:
                    if data['timestamp'] - last_timestamp < 0.075:
                        count['short'] += 1
                    else:
                        count['long'] += 1
                else:
                    print ('Invalid timestamp for data {0}'.format(msg.value()))
            print (count)
        elif msg.error().code() == KafkaError._PARTITION_EOF:
            print('End of partition reached {0}/{1}'
                  .format(msg.topic(), msg.partition()))
        else:
            print('Error occured: {0}'.format(msg.error().str()))

except KeyboardInterrupt:
    pass

finally:
    c.close()