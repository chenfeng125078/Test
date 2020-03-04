# mqtt 账号密码
# mqttSSL: ssl://10.8.200.151:8883
# mqttUserName: MQUser4Java
# mqttPwd: Mq@Jbzl$1025

# mqtt监听的主题
# vehicle/{机器人id}/routes

import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe
import ssl
import json
import os
cafile = "d:/ca.pem"
user = "MQUser4Java"
password = "Mq@Jbzl$1025"
# 外网
# server = "202.104.27.139"
# port = 18883
# 内网
server = "10.8.200.151"
port = 8883
# ssl.match_hostname = lambda cert, hostname: True
context = ssl._create_unverified_context()
auth = {'username': user, 'password': password}
d = {'ca_certs': cafile}
client = subscribe.simple("vehicle/+/routes", hostname=server,
                          port=port, auth=auth, tls=context)


def message_print(client, userdata, message):
    # print(message.topic, message.payload)
    data = message.payload.decode("utf-8")
    data = json.loads(data)  # 将string转换成json字典格式 针对内存对象 json.load()针对文件对象
    print(data)
    devicecode = data["deviceCode"]
    print(devicecode)
    with open(os.path.join("./path", (devicecode + ".json")), "w") as f:
        json.dump(data, f)


subscribe.callback(message_print, "vehicle/+/routes", hostname=server,
                   port=port, auth=auth, tls=context)
