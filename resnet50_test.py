import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

pb_file_path = './model_resnet50.pb'

# input image dimensions
img_rows, img_cols = 224, 224

img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

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

        input_x = sess.graph.get_tensor_by_name("input_2:0")
        print input_x
        out_softmax = sess.graph.get_tensor_by_name("fc1000/Softmax:0")
        print out_softmax
        out_label = sess.graph.get_tensor_by_name("output_node0:0")
        print out_label

        img_out_softmax = sess.run(out_softmax, feed_dict={input_x: x})

        # print "img_out_softmax:", img_out_softmax
        prediction_labels = np.argmax(img_out_softmax, axis=1)
        print "label:", prediction_labels
        print('Predicted:', decode_predictions(img_out_softmax, top=3)[0])