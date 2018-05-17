import tensorflow as tf
import numpy as np
from .crnn_model.crnn_model import ShadowNet
from .global_configuration import config
from .local_utils import data_utils
from .local_utils import log_utils
import matplotlib.pyplot as plt
import cv2
import os.path as ops
from PIL import Image

logger = log_utils.init_logger()

def recognize(image, weights_path, is_vis=True):
    tf.reset_default_graph()
    # image.show()
    # image = image.resize((100,32))
    # image.show()
    image = np.array(image)
    image = cv2.resize(image, (100, 32))
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = image[:, :, ::-1].copy()
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.imwrite('/Users/samueltin/Pictures/test4_cv.jpg', image)
    image = np.expand_dims(image, axis=0).astype(np.float32)


    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='input')

    phase_tensor = tf.constant('test', tf.string)
    net = ShadowNet(phase=phase_tensor, hidden_nums=256, layers_nums=2, seq_length=15,
                               num_classes=config.cfg.TRAIN.CLASSES_NUMS, rnn_cell_type='lstm')

    with tf.variable_scope('shadow'):
        net_out, tensor_dict = net.build_shadownet(inputdata=inputdata)

    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=15 * np.ones(1), merge_repeated=False)

    decoder = data_utils.TextFeatureIO()

    # config tf session
    sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    var_name_list = [v.name for v in tf.trainable_variables()]


    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        preds = sess.run(decodes, feed_dict={inputdata: image})

        preds = decoder.writer.sparse_tensor_to_str(preds[0])

        logger.info('Predict result {:s}'.format(preds[0]))

        # if is_vis:
        #     plt.figure('CRNN Model Demo')
        #     plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
        #     plt.show()

        sess.close()

        return preds[0]