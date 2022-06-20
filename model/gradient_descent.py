import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

class GradientDescent():
    def execute(self):

        X = [1., 2., 3,]
        Y = [1., 2., 3,]
        m = n_samples = len(X)
        W = tf.placeholder(tf.float32)
        hypothesis = tf.multiply(X, W)
        cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2)) / m
        W_val = []
        cost_val = []
        with tf.Session() as sess:
            #saver2 = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            #saver2.save(sess, "saved/gradient_descent")
            for i in range(-30, 50):
                W_val.append(i * 0.1)
                cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))
            plt.plot(W_val, cost_val, 'ro')
            plt.ylabel('COST')
            plt.xlabel('W')
            plt.savefig("static/img/result.svg")
            print('경사하강법 실행 중 ..')
            return "경사하강법 (Gradient Descent)"