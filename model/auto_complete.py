
import tensorflow.compat.v1 as tf
import numpy as np

class Solution:
    def __init__(self, char_arr, seq_data) -> None:
        self.num_dic = {n: i for i, n in enumerate(char_arr)}
        # ohe 을 위한 연관 배열
        # {'a':0, 'b':2, 'c':3 ....}
        self.dic_len = len(self.num_dic)
        self.optimizer = None
        self.cost = None
        self.X = None
        self.Y = None
        self.model1 = None
        self.sess = None
    
    def make_batch(self):
        
        num_dic = self.num_dic
        input_batch = []
        target_batch = []

        for seq in seq_data:
            input = [num_dic[n] for n in seq[:-1]]
            target = num_dic[seq[-1]]  # -1 은 all
            input_batch.append(np.eye(self.dic_len)[input])
            target_batch.append(target)
        return input_batch, target_batch

    def create_model(self):
        # ****
        # 옵션 설정
        # ****
        learning_rate = 0.01
        n_hidden = 128
        
        n_step = 3
        # 타입스텝: [1, 2, 3] => 3
        # RNN 을 구성하는 시퀀스의 갯수
        n_input = n_class = self.dic_len
        # 입력값 크기. 알파벳에 대한 ohe 이므로 26개가 됨
        # 따라서 c 를 선택하면 [0 0 1 0 0 0 .....0]
        # 출력값도 입력값과 마찬가지로 26개의 알파벳으로 분류합니다.
        n_input = n_class = self.dic_len

        # *******
        # 신경망 모델 구성
        # *******
        self.X = tf.placeholder(tf.float32, [None, n_step, n_input])
        self.Y = tf.placeholder(tf.int32, [None])

        W = tf.Variable(tf.random_normal([n_hidden, n_class]))
        b = tf.Variable(tf.random_normal([n_class]))

        cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        # RNN 셀 생성
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
        # 과적합 방지를 위한 Dropout 기법을 사용
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        # 여러개의 셀을 조합하기 위해 추가 셀 생성
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
        # 여러개의 셀을 조합한 RNN 셀을 생성

        outputs, states = tf.nn.dynamic_rnn(multi_cell, self.X, dtype = tf.float32)
        #  tf.nn.dynamic_rnn 을 이용해 순환 신경망을 생성

        # 최종 결과는 ohe 형식으로 생성
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]
        self.model1 = tf.matmul(outputs, W) + b

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.model1, labels=self.Y
        ))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def fit(self):
        # *******
        # 신경망 모델 학습
        # *******
        total_epoch = 30 # 훈련횟수
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        input_batch, target_batch = Solution(char_arr, seq_data).make_batch()
        for epoch in range(total_epoch):
            _, loss = self.sess.run([self.optimizer, self.cost],
                            {self.X: input_batch, self.Y: target_batch})
            print("Epoch: ", "%04d" % (epoch + 1),
                "cost: ", "{:.6f}".format(loss))
        print('===최적화 완료===')
    
    def eval(self):
        # *******
        # 신경망 모델 검증
        # *******
        prediction = tf.cast(tf.argmax(self.model1, 1), tf.int32)
        prediction_check = tf.equal(prediction, self.Y)
        # 문자열 값비교 equal
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))
        input_batch, target_batch = Solution(char_arr, seq_data).make_batch()

        predict, accuracy_val = self.sess.run([prediction, accuracy],
                                        {self.X: input_batch, self.Y: target_batch})

        predict_words = []
        for idx, val in enumerate(seq_data):
            last_char = char_arr[predict[idx]]
            predict_words.append(val[:3] + last_char)

        print('\n ===== 예측결과 ====')
        print('입력값: ', [W[:3] + ' ' for W in seq_data])
        print('예측값: ', predict_words)
        print('정확도: ', accuracy_val)

if __name__=='__main__':
    char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']

    seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']
    
    tf.disable_v2_behavior()
    s = Solution(char_arr, seq_data)
    s.create_model()
    s.fit()
    s.eval()