import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt


class Solution:
    def __init__(self) -> None:
        pass

    def hypothesis(self):
        tf.set_random_seed(777)

        X = [1, 2, 3]
        Y = [1, 2, 3]

        W = tf.placeholder(tf.float32)
        hypothesis = X * W

        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        sess = tf.Session()

        W_history = []
        cost_history = []

        for i in range(-30, 50):
            curr_W = i * 0.1
            curr_cost = sess.run(cost, {W: curr_W})
            W_history.append(curr_W)
            cost_history.append(curr_cost)
        # 차트로 확인
        plt.plot(W_history, cost_history)
        plt.show()

if __name__=='__main__':
    tf.disable_v2_behavior()
    Solution().hypothesis()