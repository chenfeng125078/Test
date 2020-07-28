# from tensorflow.keras.models import load_model
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.python.framework import graph_io
#
#
# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#     from tensorflow.python.framework.graph_util import convert_variables_to_constants
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.compat.v1.global_variables()]
#         input_graph_def = graph.as_graph_def()
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = convert_variables_to_constants(session, input_graph_def,
#                                                       output_names, freeze_var_names)
#         return frozen_graph
#
#
# """----------------------------------配置路径-----------------------------------"""
# epochs = 100
# h5_model_path = '../cnn_callback/two_kinds_model.h5'.format(epochs)
# output_path = '.'
# pb_model_name = 'my_model_ep{}.pb'.format(epochs)
#
# """----------------------------------导入keras模型------------------------------"""
# K.set_learning_phase(0)
# net_model = load_model(h5_model_path)
# print('input is :', net_model.input.name)
# print('output is:', net_model.output.name)
# # 输入节点：input_1:0
# # 输出节点：dense_3/Identity:0
#
# """----------------------------------保存为.pb格式------------------------------"""
# sess = tf.compat.v1.keras.backend.get_session()
# frozen_graph = freeze_session(sess, output_names=[net_model.output.op.name])
# graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def convert_h5to_pb():
    model = tf.keras.models.load_model("../cnn_callback/xception.h5", compile=False)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="xception.pb",
                      as_text=False)


convert_h5to_pb()
print("-------------")
