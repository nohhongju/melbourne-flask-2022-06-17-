import tensorflow.compat.v1 as tf
import numpy as np

class Solution:
    def __init__(self, seq_data) -> None:
        self.char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
        self.num_dic = {n: i for i, n in enumerate(self.char_arr)}
        self.dic_len = len(self.num_dic)
        self.optimizer = None
        self.cost = None
        self.enc_input = None
        self.dec_input = None
        self.targets = None
        self.model = None

    def make_batch(self):
        input_batch = []
        output_batch = []
        target_batch = []

        num_dic = self.num_dic
        for seq in seq_data:
            input = [num_dic[n] for n in seq[0]]
            # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다
            output = [num_dic[n] for n in ('S' + seq[1])]
            # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙인다.
            # S 는 디코더 입력의 시작
            # E 는 디코더 입력의 끝
            # P 는 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
            """
            예) 현재 배치 데이터의 최대크기가 4인 경우
            word -> ['w', 'o', 'r', 'd']
            to -> ['t', 'o', 'P', 'P']
            """
            target = [num_dic[n] for n in (seq[1] + 'E')]
            
            input_batch.append(np.eye(self.dic_len)[input])
            output_batch.append(np.eye(self.dic_len)[output])
            target_batch.append(target)
        return input_batch, output_batch, target_batch

    def create_model(self):  
        # *****
        # 옵션 설정
        # *****
        learning_rate = 0.01
        n_hidden = 128
        
        n_class = n_input = self.dic_len
        # 입력과 출력의 형태가 ohe 와 같으므로 크기도 동일함

        # *****
        # 신경망 모델 구성
        # *****

        self.enc_input = tf.placeholder(tf.float32, [None, None, n_input])
        # [배치사이즈, 타입스텝, 인풋사이즈]
        self.dec_input = tf.placeholder(tf.float32, [None, None, n_input])
        self.targets = tf.placeholder(tf.int64, [None, None]) # [배치사이즈, 타입스텝]

        # 인코더 셀을 구성
        with tf.variable_scope('encode'):
            enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.enc_input, dtype=tf.float32)
        # 디코더 셀을 구성
        with tf.variable_scope('dencode'):
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
            outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, self.dec_input,
                                                    initial_state=enc_states ,dtype=tf.float32)

        self.model = tf.layers.dense(outputs, n_class, activation=None)
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.model, labels=self.targets
        ))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def fit(self):
        total_epoch = 100

        # *****
        # 신경망 모델 학습
        # *****
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        input_batch, output_batch, target_batch = Solution().make_batch()

        for epoch in range(total_epoch):
            _, loss = self.sess.run([self.optimizer, self.cost],
                            {self.enc_input: input_batch,
                                self.dec_input: output_batch,
                                self.targets: target_batch})
            print('Epoch: ', '%04d' % (epoch + 1),'cost: ','{:6f}'.format(loss))
        print('-------최적화 완료------')

        # *****
        # 번역 테스트
        # *****
    def translate(self, word):
        seq_data = [word, 'P' * len(word)]
        input_batch, output_batch, target_batch = Solution().make_batch([])
        prediction = tf.arg_max(self.model, 2)
        result = self.sess.run(prediction,
                        {self.enc_input: input_batch,
                        self.dec_input: output_batch,
                        self.targets: target_batch})
        decoded = [self.char_arr[i] for i in result[0]]
        end = decoded.index('E')
        translated = ' '.join(decoded[:end])
        return translated

    print('======= 번역 테스트 ========')
    print('word -> ', translate('word'))
    print('love -> ', translate('love'))
    print('loev -> ', translate('loev'))
    print('girl -> ', translate('girl'))
    print('abcd -> ', translate('abcd'))

if __name__=='__main__':

    # 영어를 번역하기 위한 학습데이터
    seq_data = [['word', '단어'],['wood','나무'],
            ['game', '놀이'],['girl', '소녀'],['kiss', '키스'],['love', '사랑']]
    tf.disable_v2_behavior()
    s = Solution(seq_data)
    s.create_model()
    s.fit()
    s.translate()