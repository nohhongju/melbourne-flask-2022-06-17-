import tensorflow.compat.v1 as tf
from icecream import ic
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir


class Cabbage:
    def __init__(self):
        self.basedir = os.path.join(basedir, 'model') 
        self.df = None
        self.x_data = None
        self.y_data = None


    def preprocessing(self):
        self.df = pd.read_csv('./data/price_data.csv', encoding='UTF-8', thousands=',')
        # ic(o)
        '''
        ic| o:           year  avgTemp  minTemp  maxTemp  rainFall  avgPrice
               [[0     20100101     -4.9    -11.0      0.9       0.0      2123]
                1     20100102     -3.1     -5.5      5.5       0.8      2123
                2     20100103     -2.9     -6.9      1.4       0.0      2123
                3     20100104     -1.8     -5.1      2.2       5.9      2020
                4     20100105     -5.2     -8.7     -1.8       0.7      2060]
        '''
        # year  avgTemp  minTemp  maxTemp  rainFall  avgPrice
        # 'avgTemp', 'minTemp', 'maxTemp', 'rainFall'
        xy = np.array(self.df, dtype=np.float32)
        self.x_data = xy[:, 1:5]
        # ic(self.x_data)
        self.y_data = xy[:, 5]
        # ic(self.y_data)

    def create_model(self):  # 모델 생성
        # 텐서 모델 초기화 (모델템플릿 생성)
        model = tf.global_variables_initializer()
        # 확률변수 데이터
        self.preprocessing()
        # 선형식(가설)제작 y = Wx+b
        X = tf.placeholder(tf.float32, shape=[None, 4])  # placeholder외부에서 주입되는 값 / shape=[행, 열] - 투입되는 값
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name="weight")  # name이 반드시 있어야한다. 4개가 투입되어서 하나가 결정된다.
        b = tf.Variable(tf.random_normal([1]), name="bias")
        hypothesis = tf.matmul(X, W) + b # Wx+b hypothesis가설 -> 선형식
        # 손실함수
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        # 최적화알고리즘
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        # 세션생성
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # 트레이닝
        for step in range(100000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train],
                                        feed_dict={X: self.x_data, Y: self.y_data})
            if step % 500 == 0:
                print('# %d 손실비용: %d'%(step, cost_))
                print('- 배추가격: %d '%(hypo_[0]))
        
        # 모델저장
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt'), global_step=1000)
        print('저장완료')

    def load_model(self, avgTemp, minTemp, maxTemp, rainFall): # 모델로드
        tf.disable_v2_behavior()
        # 선형식(가설)제작 y = Wx+b
        X = tf.placeholder(tf.float32, shape=[None, 4])  # placeholder외부에서 주입되는 값 / shape=[행, 열] - 투입되는 값
        W = tf.Variable(tf.random_normal([4, 1]), name="weight")  # name이 반드시 있어야한다. 4개가 투입되어서 하나가 결정된다.
        b = tf.Variable(tf.random_normal([1]), name="bias")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt-1000'))
            data = [[avgTemp,minTemp,maxTemp,rainFall],]
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})
            print(dict)
        return int(dict[0])

if __name__=='__main__':
    tf.disable_v2_behavior()
    Cabbage().create_model()