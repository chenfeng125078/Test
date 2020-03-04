# application/x-www-form-urlencoded格式 (默认填充格式)
# import requests
#
# datas = {"destination": {"buildingName": None,
#                          "buildingNum": None,
#                          "deviceCode": None,
#                          "floorNum": "5",
#                          "state": None,
#                          "typeCode": None,
#                          "x": "6807.5",
#                          "y": "6774.9",
#                          "yaw": None,
#                          "z": None}, "deviceCode": "robotcar1", "orderId": "f138fe1f-7768-4850-975e-23fe78603fe9",
#          "route": None, "startFloorNum": "5"}
# r = requests.post("http://10.8.202.166:65200/v1/tcs/request/createTransportOrder", data=datas)
# print(r.text)
# print(r.status_code)

"""-----------------------"""
# application/json格式
# 3个可用点 (9149,2800) (32856,2200) (23300,8549),(4500,9795)
# 下单操作时: 暂时不能跨层,订单号需修改

import json
import requests
headers = {'Content-Type': 'application/json'}
datas = json.dumps({"destination": {"buildingName": None,
                                    "buildingNum": None,
                                    "deviceCode": None,
                                    "floorNum": "7",
                                    "state": None,
                                    "typeCode": None,
                                    "x": "23300.0",
                                    "y": "8549.0",
                                    "yaw": None,
                                    "z": None}, "deviceCode": "Vehicle-0002",
                    "orderId": "1234-56780000003",
                    "route": None, "startFloorNum": "7"})
r = requests.post("http://10.8.202.166:65200/v1/tcs/request/createTransportOrder", data=datas, headers=headers)
print(r.text)

"-----------------------"
# text/xml数据格式

# import requests
# headers = {"Content-Type": "text/xml"}
# datas = """<?xml version="1.0"?>
# <methodCall>
#     <methodName>examples.getStateName</methodName>
#     <params>
#         <param>
#             <value><i4>41</i4></value>
#         </param>
#     </params>
# </methodCall>"""
# r = requests.post("http://httpbin.org/post", data=datas, headers=headers)
# print(r.text)

"----------------------"
# multipart/form-data数据格式 (文件上传)

# import requests
# files = {"file": open("C:/Users/Administrator/Desktop/test.txt", "rb")}
# r = requests.post("http://httpbin.org/post", files=files)
# print(r.text)
