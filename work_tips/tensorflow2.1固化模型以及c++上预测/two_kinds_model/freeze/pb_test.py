import tensorflow as tf
import os
import cv2
import numpy as np


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():

        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])

    import_graph = wrapped_import.graph

    print("-" * 50)

    print("Frozen model layers: ")

    layers = [op.name for op in import_graph.get_operations()]

    if print_graph:
        for layer in layers:
            print(layer)

    print("-" * 50)

    return wrapped_import.prune(

        tf.nest.map_structure(import_graph.as_graph_element, inputs),

        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def test_one_image(image_path):
    # 测试数据集，
    # (train_images, train_labels), (test_images,
    #
    #                                test_labels) = tf.keras.datasets.mnist.load_data()
    # image_path = os.path.join(".", "45.bmp")
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    image = cv2.resize(image, (128, 128))
    # print(image)
    image = np.array(image, dtype=np.float32)
    print(image)
    test_image = (image.reshape(-1, 128, 128, 3) / 255.0)

    # Load frozen graph using TensorFlow 1.x functions

    with tf.io.gfile.GFile("./frozen_models/test.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,

                                    inputs=["Input:0"],

                                    outputs=["Identity:0"],

                                    print_graph=True)

    print("-" * 50)

    print("Frozen model inputs: ")

    print(frozen_func.inputs)

    print("Frozen model outputs: ")

    print(frozen_func.outputs)

    # Get predictions for test images

    predictions = frozen_func(Input=tf.constant(test_image))

    # Print the prediction for the first image

    print("-" * 50)

    print("Example prediction reference:")
    print(predictions[0])
    print(predictions[0].numpy())
    label_dict = {0: "晶典", 1: "污渍"}
    label = int(np.argmax(predictions[0].numpy()))
    label = label_dict[label]

    print("-" * 50)
    print(label)


if __name__ == "__main__":
    test_one_image("./45.bmp")
