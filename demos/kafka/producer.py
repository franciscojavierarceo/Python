import argparse
import json
import time

from confluent_kafka import Producer

code = {
    'A': '*-',
    'B': '-***',
    'C': '-*-*',
    'D': '-**',
    'E': '*',
    'F': '**-*',
    'G': '--*',
    'H': '****',
    'I': '**',
    'J': '*---',
    'K': '-*-',
    'L': '*-**',
    'M': '--',
    'N': '-*',
    'O': '---',
    'P': '*--*',
    'Q': '--*-',
    'R': '*-*',
    'S': '***',
    'T': '-',
    'U': '**-',
    'V': '***-',
    'W': '*--',
    'X': '-**-',
    'Y': '-*--',
    'Z': '--**',
}

def get_json_str(timestamp, new_status):
    d = {
        'timestamp': timestamp,
        'new_status': new_status,
    }
    print (json.dumps(d))
    return json.dumps(d)

def send_long(p, topic, key):
    p.produce(topic, key=key, value=get_json_str(time.time(), "on"))
    time.sleep(0.1)
    p.produce(topic, key=key, value=get_json_str(time.time(), "off"))
    time.sleep(0.05)

def send_short(p, topic, key):
    p.produce(topic, key=key, value=get_json_str(time.time(), "on"))
    time.sleep(0.05)
    p.produce(topic, key=key, value=get_json_str(time.time(), "off"))
    time.sleep(0.05)

def send_letter(letter, p, topic, key):
    for send in code[letter]:
        if send == '*':
            send_short(p, topic, key)
        else:
            send_long(p, topic, key)
    time.sleep(0.1)

parser = argparse.ArgumentParser(description='Send string via morse code light bulb.')
parser.add_argument('--key', type=str, default='1',
                    help='key')
parser.add_argument('--topic', type=str, default='light_bulb',
                    help='publish topic')
parser.add_argument('--string', type=str, default='ABC',
                    help='string data (A-Z)')

args = parser.parse_args()
if not args.string.isalpha():
    raise RuntimeError("Input string should only contain letter A-Z.")

p = Producer({'bootstrap.servers': 'localhost:9091'})
for letter in args.string.upper():
    send_letter(letter, p, args.topic, args.key)
p.flush(30)