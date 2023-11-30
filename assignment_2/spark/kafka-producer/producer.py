import json
import threading

from kafka import KafkaProducer


def kafka_python_producer_sync(producer, msg, topic):
    producer.send(topic, bytes(msg, encoding='utf-8'))
    print("Sending " + msg)
    producer.flush(timeout=60)


def success(metadata):
    print(metadata.topic)


def error(exception):
    print(exception)


def kafka_python_producer_async(producer, msg, topic):
    producer.send(topic, bytes(msg, encoding='utf-8')).add_callback(success).add_errback(error)
    producer.flush()


def produce_from_file(producer, file):
    print(file)
    with open(file, encoding='utf-8') as f:
        for line in f:
            # Load each line separately as a JSON object
            try:
                item = json.loads(line)
                kafka_python_producer_sync(producer, json.dumps(item), 'mock')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {line}. {e}")



def run_job():
    producer = KafkaProducer(bootstrap_servers='34.71.12.54:9092')  # use your VM's external IP Here!
    # Change the path to your laptop!
    # if you want to learn about threading in python, check the following article
    # https://realpython.com/intro-to-python-threading/
    # if you want to schedule a job https://www.geeksforgeeks.org/python-schedule-library/
    t1 = threading.Thread(target=produce_from_file,
                          args=(producer, r'C:\Users\georg\PycharmProjects\DE_Group11\assignment_2\spark\data\house_pricing_sample_1.json_part-00000-2fa7462b-699c-4bd4-82ea-cbb3a570ff44-c000.json',))
    t2 = threading.Thread(target=produce_from_file,
                          args=(producer, r'C:\Users\georg\PycharmProjects\DE_Group11\assignment_2\spark\data\house_pricing_sample_2.json_part-00000-ae3cda91-5dac-49eb-86b1-fcf5cb79a18e-c000.json',))
    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == '__main__':
    run_job()
