import numpy as np


if __name__ == '__main__':
    order_id = ""
    i = np.random.randint(0, 14)
    for item in range(i):
        j = np.random.randint(0, 10)
        order_id += str(j)
    print(order_id)
