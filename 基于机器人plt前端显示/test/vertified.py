import paho.mqtt.client as mqtt
import ssl
cafile = "d:/ca.pem"
user = "MQUser4TCS"
passwd = "Mq@Tbzl$1025"
server = "202.104.27.139"
port = 18883
ssl.match_hostname = lambda cert, hostname: True


def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # client.subscribe("$SYS/broker/version")
    client.subscribe("+/#")


def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))


client = mqtt.Client()
client.username_pw_set(user, passwd)
client.on_connect = on_connect
client.on_message = on_message
client.tls_set(cafile)
client.connect(server, port, 60)
client.loop_forever()
