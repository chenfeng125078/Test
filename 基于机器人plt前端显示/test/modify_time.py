# import os
# import sys
# import time
#
#
# path = os.path.join(".", "path")
# print(path)
# current_time = time.time()
# # print(time.localtime(current_time))
# for item in os.listdir(path):
#     the_time = os.stat(os.path.join(path, item)).st_mtime
#     # print("%s,%f" % (item, modify_time))
#     modify_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(the_time))
#     if current_time - the_time < 1200:
#         print("-----------", item)
#     print("%s,%s" % (item, modify_time))
