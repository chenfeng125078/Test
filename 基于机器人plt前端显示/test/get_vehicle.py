import urllib
import urllib.request
# import threading
import json
import time
import os
from watchdog.observers import Observer
from watchdog.events import *


def receive_position():
    url = "http://10.8.202.166:65200/v1/tcs/vehicles"
    req = urllib.request.Request(url)
    # print(req)
    res_data = urllib.request.urlopen(req)
    res = res_data.read()
    # print(res)
    data = json.loads(res)
    # print(data)
    # print("-------------")
    for item in data:
        # print(item)
        if item:
            write_case = False
            devicecode = item["DeviceCode"]
            # print(devicecode)
            if not os.path.exists(os.path.join(".", "data", devicecode + ".json")):
                with open(os.path.join(".", "data", devicecode + ".json"), "w") as fp:
                    print("--write--")
                    json.dump(item, fp)
            with open(os.path.join(".", "data", devicecode + ".json"), "r") as fp:
                last_data = json.load(fp)
                # print("----------")
                if last_data["Location-x"] == item["Location-x"] and last_data["Location-y"] == item["Location-y"]:
                    continue
                else:
                    write_case = True
            if write_case:
                with open(os.path.join(".", "data", devicecode + ".json"), "w") as fp:
                    print("--write--")
                    json.dump(item, fp)
            # time.sleep(1)


class PathFileMonitorHandler(FileSystemEventHandler):
    def __init__(self, **kwargs):
        super(PathFileMonitorHandler, self).__init__(**kwargs)
        # 监控目录 目录下面以device_id为目录存放各自的图片
        self._watch_path = "./path"
        self.case = False

    # 重写文件改变函数，文件改变都会触发文件夹变化
    def on_modified(self, event):
        if not event.is_directory:
            file_path = event.src_path
            print("文件改变: %s " % file_path)
            with open(file_path, "r") as fp:
                data = json.load(fp)
                print(data["Location-x"])
            self.case = True

    # def on_created(self, event):
    #     print('创建了文件夹', event.src_path)
    #
    # def on_moved(self, event):
    #     print("移动了文件", event.src_path)
    #
    # def on_deleted(self, event):
    #     print("删除了文件", event.src_path)


def monitor_file():
    observer.schedule(event_handler, path="./data", recursive=True)  # recursive递归的
    observer.start()
    for i in range(120):
        receive_position()
        print(i)
        time.sleep(1)
    print("----complete----")
    observer.join(3)
    # if event_handler.case:
    #     # observer.stop()
    #     return True
    # else:
    #     return False


if __name__ == '__main__':
    event_handler = PathFileMonitorHandler()
    observer = Observer()
    monitor_file()
