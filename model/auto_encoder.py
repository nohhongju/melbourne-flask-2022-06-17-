import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

class Solution:
    def __init__(self) -> None:
        mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

        # ******
        # 옵션 설정
        # ******
        learning_rate = 0.01
        training_epoch = 20
        batch_size = 100
        # 신경망 레이어 구성옵션
        n_hidden = 256 # 히든 레이어의 뉴런 갯수
        n_input = 28 * 28
        
        # ******
        # 신경망 모델 구성
        # ******
        X = tf.placeholder(tf.float32, [None, n_input])
        # Y 는 선언하지 않음. 입력값을 Y 로 사용하기 때문

        W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
        b_encode = tf.Variable(tf.random_normal([n_hidden]))
        # 인코더 레이어와 디코더 레이어의 가중치와 편향변수를 설정

        encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

        W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
        b_decode = tf.Variable(tf.random_normal([n_input]))
        # 디코더 레이어 구성. 이 디코더가 최종 모델이 됨
        decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

        # 디코더는 인풋과 최대한 같은 결과를 내야하므로, 디코딩한 결과를 평가하기 위해
        # 입력 값이 X 값을 평가를 위한 실측 결과 값으로 하여 decoder 와의
        # 차이를 손실값으로 설정합니다.

        cost = tf.reduce_mean(tf.pow(X - decoder, 2))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        # ******
        # 신경망 모델 학습
        # ******
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        total_batch = int(mnist.train.num_examples/batch_size)
        for epoch in range(training_epoch):
            total_cost = 0

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, cost_val = sess.run([optimizer, cost],
                                    {X: batch_xs})
                total_cost += cost_val
            print('Epoch: ','%04d' % (epoch + 1),
                'Avg Cost: ','{:.4f}'.format(total_cost / total_batch))

        print('-----최적화 완료------')

        # ******
        # 신경망 모델 테스트(검정)
        # ******
        sample_size = 10
        samples = sess.run(decoder, {X: mnist.test.images[:sample_size]})

        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()
            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28))) # mnist 사이즈가 28
            ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

        plt.show()