import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2

pb_file_path = './mobilenet_multiregression.pb'

# input image dimensions
img_rows, img_cols = 224, 224

img_path = 'dog.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (224,224))
x = np.expand_dims(img, axis=0)

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open(pb_file_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        op = sess.graph.get_operations()
        tensor_name = [m.values() for m in op]
        print tensor_name

        input_x = sess.graph.get_tensor_by_name("input_1:0")
        print input_x
        out_score0 = sess.graph.get_tensor_by_name("output_node0:0")
        out_score11 = sess.graph.get_tensor_by_name("output_node11:0")
        # print out_score

        img_out_softmax = sess.run([out_score0, out_score11], feed_dict={input_x: x})

        print "img_out_softmax:", img_out_softmax