import tensorflow as tf
# from create_tf_record import *
from tensorflow.python.framework import graph_util


def freeze_graph(input_checkpoint,output_graph):
    output_node_names = "decoder/output"
    saver = tf.train.import_meta_graph(input_checkpoint + ".meta", clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess,input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(",")
        )
        
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    input_checkpoint = "kittiseg_road_detection/RUNS/zhihui/model.ckpt-11999"
    output_pb_path = "kittiseg_road_detection/RUNS/zhihui/frozen_model.pb"
    freeze_graph(input_checkpoint, output_pb_path)
