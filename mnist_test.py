import tensorflow as tf
import numpy as np
from keras.datasets import mnist

pb_file_path = './model.pb'

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
_, (X_test, y_test) = mnist.load_data()

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_test = X_test.astype('float32')
X_test /= 255
print(X_test.shape[0], 'test samples')

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
        # print tensor_name

        input_x = sess.graph.get_tensor_by_name("conv2d_1_input:0")
        print input_x
        out_softmax = sess.graph.get_tensor_by_name("activation_4/Softmax:0")
        print out_softmax
        out_label = sess.graph.get_tensor_by_name("output_node0:0")
        print out_label

        img_out_softmax = sess.run(out_softmax, feed_dict={input_x: X_test[0:5]})

        print "img_out_softmax:", img_out_softmax
        prediction_labels = np.argmax(img_out_softmax, axis=1)
        print "gt label:", y_test[0:5]
        print "label:", prediction_labels