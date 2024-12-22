from __future__ import division, print_function, absolute_import
import os
import argparse
import datetime
import matplotlib
import numpy as np
import scipy.io
import csv
from sklearn.decomposition import NMF
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf2

tf = tf2.compat.v1
tf.disable_v2_behavior()
# 跟原来的在理论上没有区别
# 矩阵填充模型



from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use('Agg')

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='new-model')

# Print debug statements
parser.add_argument('--debug', type=bool, default=True, nargs='+', help="Want debug statements.")
parser.add_argument('--debug_display_step', type=int, default=1, help="Display loss after.")

# Hyper-parameters
parser.add_argument('--hidden_units', type=int, default=1000, help="Size of hidden layer or latent space dimensions.")
parser.add_argument('--lambda_val', type=int, default=1000,
                    help="Regularization coefficient, to control the contribution of regularization term in the cost function.")
parser.add_argument('--initial_learning_rate', type=float, default=0.0001, help="Initial value of learning rate.")
parser.add_argument('--iterations', type=int, default=30000, help="Number of iterations to train the model for.")
# 在连续迭代中损失函数值的变化小于阈值之后停止梯度下降，这意味着收敛。
parser.add_argument('--threshold', type=int, default=0.0001,
                    help="To stop gradient descent after the change in loss function value in consecutive iterations is less than the threshold, implying convergence.")

# Data
parser.add_argument('--data', type=str, default='75748.csv',
                    help="Dataset to run the script on. In the paper we choose from : ['52529.csv', '75748.csv', '74672.csv', 'Simulate_v9.csv', 'Simulate_v10.csv']")

# Model save and restore options
parser.add_argument('--save_model_location', type=str, default='checkpoints/admodel1.ckpt',
                    help="Location to save the learnt model")
parser.add_argument('--load_model_location', type=str, default='checkpoints/admodel0.ckpt',
                    help="Load the saved model from.")
parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")
parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")

# masked and imputed matrix save location / name
parser.add_argument('--imputed_save', type=str, default='ad74672_2out1', help="save the imputed matrix as")
parser.add_argument('--masked_save', type=str, default='adout', help="save the masked matrix as")

FLAGS = parser.parse_args()

if __name__ == '__main__':
    # started clock
    start_time = datetime.datetime.now()

    if not os.path.exists('checkpoints1'):
        os.makedirs('checkpoints1')

    # if FLAGS.debug:
    #     if not FLAGS.load_saved:
    #         with open(FLAGS.log_file, 'w') as log:
    #             log.write('Step\tLoss\tLoss per Cell\t Change \n')

    # reading dataset
    try:
        extn = FLAGS.data.split('.')[1]
        if (extn == 'mat'):
            print("[!data read] Reading from data/" + FLAGS.data)
            processed_count_matrix = scipy.io.mmread("data/" + FLAGS.data)
            processed_count_matrix = processed_count_matrix.toarray()
            processed_count_matrix = np.array(processed_count_matrix)
        else:
            print("[!data read] Reading from data/" + FLAGS.data)
            with open("data/" + FLAGS.data) as f:
                ncols = len(f.readline().split(','))
            processed_count_matrix = np.loadtxt(open("data/" + FLAGS.data, "rb"), delimiter=",", skiprows=1,
                                                usecols=range(1, ncols + 1))
    except:
        print(
            "[!data read] Please make sure that your processed dataset is in data/ in .mat or .csv format and you have entered the filename as data parameter. e.g. 52529.csv")
        exit()

    dataset = FLAGS.data.split('.')[0]


    # finding number of genes and cells.记录细胞数和基因数
    # 基因数
    genes = 100
    # 细胞数
    cells = processed_count_matrix.shape[0]
    print("[info] Genes : {0}, Cells : {1}".format(genes, cells))
    mask_index = np.where(processed_count_matrix == 0)

    model = NMF(n_components=100, init='random', random_state=0, max_iter=1000)
    U = model.fit_transform(processed_count_matrix)
    V = model.components_
    print(U)

    # placeholder definitions
    X = tf.placeholder("float32", [None, genes])
    mask = tf.placeholder("float32", [None, genes])

    matrix_mask = U.copy()
    # 将矩阵中所有非0元素赋值成1
    matrix_mask[U.nonzero()] = 1

    print("[info] Hyper-parameters")
    print("\t Hidden Units : " + str(FLAGS.hidden_units))
    print("\t Lambda : {0}".format(FLAGS.lambda_val))
    print("\t Threshold : " + str(FLAGS.threshold))
    print("\t Iterations : " + str(FLAGS.iterations))
    print("\t Initial learning rate : " + str(FLAGS.initial_learning_rate))

    # model definition
    weights = {
        # 编码器权重
        'encoder_h': tf.Variable(tf.random_normal([genes, FLAGS.hidden_units])),
        # 解码器权重
        'decoder_h': tf.Variable(tf.random_normal([FLAGS.hidden_units, genes])),
    }
    biases = {
        # 编码器偏差
        'encoder_b': tf.Variable(tf.random_normal([FLAGS.hidden_units])),
        # 解码器偏差
        'decoder_b': tf.Variable(tf.random_normal([genes])),
    }


    def encoder(x):
        # 编码器
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h']), biases['encoder_b']))
        return layer_1


    def decoder(x):
        # 解码器
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h']), biases['decoder_b']))
        return layer_1


    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # loss definition
    y_pred = decoder_op
    y_true = X

    rmse_loss = tf.pow(tf.norm(y_true - y_pred * mask), 2)

    regularization = tf.multiply(tf.constant(FLAGS.lambda_val / 2.0, dtype="float32"),
                                 tf.add(tf.pow(tf.norm(weights['decoder_h']), 2),
                                        tf.pow(tf.norm(weights['encoder_h']), 2)))
    # regularization = 0
    loss = tf.add(tf.reduce_mean(rmse_loss), regularization)
    optimizer = tf.train.RMSPropOptimizer(FLAGS.initial_learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if (FLAGS.load_saved):
            saver.restore(sess, FLAGS.load_model_location)
            print("[info] model restored.")
        else:
            sess.run(init)
        prev_loss = 0
        for k in range(1, FLAGS.iterations + 1):
        # for k in range(1, 3):
            _, loss = sess.run([optimizer, rmse_loss], feed_dict={X: U, mask: matrix_mask})
            lpentry = loss / cells
            change = abs(prev_loss - lpentry)
            if (change <= FLAGS.threshold):
                print("Reached the threshold value.")
                print(change)
                break
            prev_loss = lpentry
            if (FLAGS.debug):
                if (k - 1) % FLAGS.debug_display_step == 0:
                    print('Step %i : Total loss: %f, Loss per Cell : %f, Change : %f' % (k, loss, lpentry, change))
                    with open(FLAGS.log_file, 'a') as log:
                        log.write('{0}\t{1}\t{2}\t{3}\n'.format(
                            k,
                            loss,
                            lpentry,
                            change
                        ))
            if ((k - 1) % 5 == 0):
                save_path = saver.save(sess, FLAGS.save_model_location)

        imputed_count_matrix = sess.run([y_pred], feed_dict={X: U, mask: matrix_mask})
        imputed_count_matrix = np.array(imputed_count_matrix)
        imputed_count_matrix = imputed_count_matrix.reshape(imputed_count_matrix.shape[1], -1)
        print(imputed_count_matrix)
        imputed_count_matrix = np.dot(imputed_count_matrix, V)
        print(imputed_count_matrix)
        print(imputed_count_matrix.shape)
        processed_count_matrix[mask_index] = imputed_count_matrix[mask_index]



        # scipy.io.savemat(FLAGS.imputed_save + "1111.mat", mdict = {"arr" : imputed_count_matrix})
        with open('75748_new.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(processed_count_matrix)

    finish_time = datetime.datetime.now()
    print("[info] Total time taken = {0}".format(finish_time - start_time))
