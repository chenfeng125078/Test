import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe
import ssl
import json
import os
import time
import matplotlib.pyplot as plt
import cv2

cafile = "d:/ca.pem"
user = "MQUser4TCS"
passwd = "Mq@Tbzl$1025"
# 外网
# server = "202.104.27.139"
# port = 18883
# 内网服务器
server = "10.8.200.151"
# 内网端口
port = 8883
# ssl.match_hostname = lambda cert, hostname: True
context = ssl._create_unverified_context()
auth = {'username': user, 'password': passwd}
d = {'ca_certs': cafile}

client = subscribe.simple("device/+/deviceinfo", hostname=server,
                          port=port, auth=auth, tls=context)


def on_message_print(client, userdata, msg):
    # time.sleep(0.5)
    data = msg.payload.decode('utf-8')
    data = json.loads(data)
    # print("---------")
    devicecode = msg.topic.split('/')[1]
    data['deviceCode'] = devicecode
    print(devicecode)
    if not os.path.exists(os.path.join(".", "data", devicecode + ".json")):
        with open(os.path.join(".", "data", devicecode + ".json"), "w") as fp:
            print("--write--")
            json.dump(data, fp)
    # 新旧数据进行对比,来决定文件是否重写
    with open(os.path.join(".", "data", devicecode + ".json"), "r") as fp:
        last_data = json.load(fp)
        # print("----------")
        if last_data["Location-x"] == data["Location-x"] and last_data["Location-y"] == data["Location-y"]:
            pass
        else:
            write_case = True
    if write_case:
        with open(os.path.join(".", "data", devicecode + ".json"), "w") as fp:
            print("--write--")
            json.dump(data, fp)


subscribe.callback(on_message_print, "device/+/deviceinfo", hostname=server,
                   port=port, auth=auth, tls=context)
