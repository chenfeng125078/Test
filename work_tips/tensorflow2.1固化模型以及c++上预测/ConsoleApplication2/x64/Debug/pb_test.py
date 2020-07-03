import tensorflow as tf
import os
import cv2
import numpy as np
import glob
# import time


class Classify(object):
    def __init__(self):
        self.label_dict = {0: "晶典", 1: "污渍"}
        self.path = "D:\\test.pb"
        self.load_model()

    def load_model(self):
        def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
            def _imports_graph_def():

                tf.compat.v1.import_graph_def(graph_def, name="")

            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])

            import_graph = wrapped_import.graph

            # print("-" * 50)
            #
            # print("Frozen model layers: ")

            layers = [op.name for op in import_graph.get_operations()]

            # if print_graph:
            #     for layer in layers:
            #         print(layer)
            #
            # print("-" * 50)

            return wrapped_import.prune(

                tf.nest.map_structure(import_graph.as_graph_element, inputs),

                tf.nest.map_structure(import_graph.as_graph_element, outputs))

        with tf.io.gfile.GFile(self.path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        # Wrap frozen graph to ConcreteFunctions
        self.frozen_func = wrap_frozen_graph(graph_def=graph_def,

                                             inputs=["Input:0"],

                                             outputs=["Identity:0"],

                                             print_graph=True)
        # print("-" * 50)
        print("Frozen model inputs: ")
        print(self.frozen_func.inputs)
        print("Frozen model outputs: ")
        print(self.frozen_func.outputs)
        img = cv2.imdecode(np.fromfile("D:\\45.bmp", dtype=np.uint8), 1)
        self.recognize(img)
        # print("++++++++++++++++++++++++++++++++")

    def recognize(self, image):
        try:
            # print("---------------------recognize:------------------------")
            # print(type(image))
            # print(dir(image))
            image = np.array(image, dtype=np.float32)
            # cv2.imwrite("1.bmp", image)
            # print("-------------", image_path)
            # image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
            # print("1")
            image = cv2.resize(image, (128, 128))
            # print("2")
            test_image = (image.reshape(-1, 128, 128, 3) / 255.0)
            predictions = self.frozen_func(Input=tf.constant(test_image))

            # end_time = time.time()
            # print("use time:", end_time - start_time)
            # print("3")
            # Print the prediction for the first image
            # print("-" * 50)
            # print("Example prediction reference:")
            # print("----------", predictions[0])
            # print(predictions[0].numpy())
            label = int(np.argmax(predictions[0].numpy()))
            label = self.label_dict[label]
            print(label)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    class_1 = Classify()
    images = glob.glob(os.path.join("D:\\data\\test_2", "*.bmp"))
    # print("model load complete !")
    for item in images:
        img = cv2.imdecode(np.fromfile(item, dtype=np.uint8), 1)
        # print("=================")
        class_1.recognize(img)
