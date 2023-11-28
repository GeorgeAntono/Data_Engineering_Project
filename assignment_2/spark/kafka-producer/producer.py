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
    f = open(file)
    data = json.load(f)
    for item in data:
        kafka_python_producer_sync(producer, json.dumps(item), 'mock')
    f.close()


def run_job():
    producer = KafkaProducer(bootstrap_servers='35.225.176.195:9092')  # use your VM's external IP Here!
    # Change the path to your laptop!
    # if you want to learn about threading in python, check the following article
    # https://realpython.com/intro-to-python-threading/
    # if you want to schedule a job https://www.geeksforgeeks.org/python-schedule-library/
    t1 = threading.Thread(target=produce_from_file,
                          args=(producer, 'C://Users/georg/PycharmProjects/DE_Group11/assignment_2/spark/data/MOCK_DATA.json',))
    t2 = threading.Thread(target=produce_from_file,
                          args=(producer, 'C://Users/georg/PycharmProjects/DE_Group11/assignment_2/spark/data/MOCK_DATA_NULL.json',))
    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == '__main__':
    run_job()
