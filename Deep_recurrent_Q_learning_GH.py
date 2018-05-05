import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import cv2
import time
from tensorflow.contrib.layers import flatten
from tensorflow.contrib import rnn
# Import games
sys.path.append("Wrapped_Game/")

import pong as game

action_size = 3
EPISODE = 20000

episode = 1
load = False
epsilon = 1
epsilon_start = 1
epsilon_end = 0.05
epsilon_step = 120000
learning_rate = 0.0001
update_target_rate = 1000-1#################################
memory_size = 20000
state_size = [80, 80, 1]
stack_size = 1

replay_memory = []
decay_rate = 0.99
skip_size = 3

plot_episode = 300

train_start = 19999
h_size = 128
train_step=8
batch_size = 8


first_conv   = [8,8,1,32]
second_conv  = [4,4,32,64]
third_conv   = [3, 3, 64, 64]
first_dense  = [10 * 10 * 64, 1024]
second_dense = [1024, h_size]
third_dense  = [256, action_size]



def initializer(shape):

    dim_sum = np.sum(shape)

    if len(shape) == 1:
        dim_sum += 1

    bound = np.sqrt(2.0 / dim_sum)
    return tf.random_uniform(shape, minval=-bound, maxval=bound)

def weight(shape):

    return tf.Variable(initializer(shape)), tf.Variable(initializer([shape[-1]]))


def conv2d(x, w, stride):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')


def pre_processing(image):
    image = cv2.resize(image,(80,80))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = np.reshape(image,(80,80,1))

    image = (image - (255.0/2))/(255.0/2)

    image = np.uint8(image)

    return image

image = tf.placeholder(tf.float32, shape=[None, state_size[0], state_size[1],
                                                       state_size[2] * stack_size])

class build_model:

    def __init__(self, cell, myscope):


        self.w_conv1, self.b_conv1 = weight(first_conv)
        self.w_conv2, self.b_conv2 = weight(second_conv)
        self.w_conv3, self.b_conv3 = weight(third_conv)
        self.w_dense1, self.b_dense1 = weight(first_dense)
        self.w_dense2, self.b_dense2 = weight(second_dense)
        self.w_dense3, self.b_dense3 = weight(third_dense)


        h_conv1 = tf.nn.relu(conv2d(image, self.w_conv1, 4) + self.b_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_conv1, self.w_conv2, 2) + self.b_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_conv2, self.w_conv3, 1) + self.b_conv3)

        h_pool3_flat = tf.reshape(h_conv3, [-1, first_dense[0]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, self.w_dense1) + self.b_dense1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.w_dense2) + self.b_dense2)

        self.batch_size = tf.placeholder(tf.int32,[])
        self.trainLength = tf.placeholder(dtype=tf.int32)

        # 콘볼루션의 마지막 부분 1x1x512 를 512 로 바꾸고, 이를 batch x trace x hidden node 로 바꿈

        convFlat = tf.reshape(flatten(h_fc2), [self.batch_size, self.trainLength, h_size])   ##action 뽑을 때는 batch랑 trainLength 가 1,1 ######################
        # rnn hidden node의 초기 상태를 0으로 초기화

        self.state_in = cell.zero_state(self.batch_size, tf.float32)
        #print('state_size',self.state_in)

        # dynamic_rnn 을 이용해 rnn output과 다음 상태를 반환함
        rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=convFlat, cell=cell, dtype=tf.float32, initial_state=self.state_in, scope=myscope+'_rnn')
        rnn = tf.reshape(rnn, shape=[-1, h_size])

        self.VW = tf.Variable(tf.random_normal([int(h_size), 3]),dtype=tf.float32)

        self.output = tf.matmul(rnn, self.VW)
        #print('output_size',np.shape(self.output))


cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size, state_is_tuple=True)
cellT = tf.nn.rnn_cell.LSTMCell(num_units=h_size, state_is_tuple=True)

model = build_model(cell,'model')
target_model = build_model(cellT,'target')

def target_update():
    update_wc1 = tf.assign(target_model.w_conv1, model.w_conv1)
    update_wc2 = tf.assign(target_model.w_conv2, model.w_conv2)
    update_wc3 = tf.assign(target_model.w_conv3, model.w_conv3)
    update_bc1 = tf.assign(target_model.b_conv1, model.b_conv1)
    update_bc2 = tf.assign(target_model.b_conv2, model.b_conv2)
    update_bc3 = tf.assign(target_model.b_conv3, model.b_conv3)
    update_wd1 = tf.assign(target_model.w_dense1, model.w_dense1)
    update_wd2 = tf.assign(target_model.w_dense2, model.w_dense2)
    update_wd3 = tf.assign(target_model.w_dense3, model.w_dense3)
    update_bd1 = tf.assign(target_model.b_dense1, model.b_dense1)
    update_bd2 = tf.assign(target_model.b_dense2, model.b_dense2)
    update_bd3 = tf.assign(target_model.b_dense3, model.b_dense3)
    update_rnn_vw = tf.assign(target_model.VW,model.VW)

    sess.run(update_wc1)
    sess.run(update_wc2)
    sess.run(update_wc3)
    sess.run(update_bc1)
    sess.run(update_bc2)
    sess.run(update_bc3)
    sess.run(update_wd1)
    sess.run(update_wd2)
    sess.run(update_wd3)
    sess.run(update_bd1)
    sess.run(update_bd2)
    sess.run(update_bd3)
    sess.run(update_rnn_vw)

    trainable = tf.trainable_variables()
    model_var = [var for var in trainable if var.name.startswith('model_rnn')]
    #print('model_var',model_var)
    target_var = [var for var in trainable if var.name.startswith('target_rnn')]
    #print('target_var',target_var)

    for i in range(len(model_var)):
        update_rnn = tf.assign(target_var[i],model_var[i])
        sess.run(update_rnn)

    #target_var = model_var

    #update_rnn = tf.assign(target_var, model_var)
    #sess.run(update_rnn)




def get_sample(batch_size,time_step):

    sampledTraces=[]
    for i in range(batch_size):
        point = np.random.randint(0,len(replay_memory)+1-time_step)
        sampledTraces.append(replay_memory[point:point+time_step])

    return np.reshape(sampledTraces, [batch_size * time_step, 5])
     ### 근데 이러면 24개중에 32개 뽑는거라서 겹치는게 반드시 생긴다..





# 타겟과 예측 Q value 사이의 차이의 제곱합이 손실이다.
# 타겟Q를 받는 부분
y_prediction = tf.placeholder(shape=[None], dtype=tf.float32)
# 행동을 받는 부분
action_target = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
# 행동을 one_hot 인코딩 하는 부분
# self.actions_onehot = tf.one_hot(self.actions, 3, dtype=tf.float32)

# 각 네트워크의 행동의 Q 값을 골라내는 것
y_net = tf.reduce_sum(tf.multiply(model.output, action_target), reduction_indices=1)

# 각각의 차이
td_error = tf.square(y_prediction - y_net)

# 신경망을 통해 정확한 그라디언트만 보내기 위해, Lample & Chatlot 2016 에서 각 기록에 대한 손실의 첫 절반을 마스크 할 것이다.
maskA = tf.zeros([batch_size, train_step/2])  # 4는 trace 의 수
maskB = tf.ones([batch_size, train_step/2])
mask = tf.concat([maskA, maskB],1)
mask = tf.reshape(mask, [-1])

# 뒤에 절반만 가지고 손실을 계산한다
loss = tf.reduce_mean(td_error * mask)

trainer = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.01)
# 업데이트 함수
updateModel = trainer.minimize(loss)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()

agent = game.GameState()


step = 0
global_step = 0
Image=[]
score = 0

plt.figure(1)
plot_x = []
plot_y = []

terminal = True
print('start')
while(terminal):
    if global_step < train_start:
        states = 'Observing'

        action = np.zeros([action_size])

        if len(replay_memory)==0:
            observe,_,_ = agent.frame_step(action)
            state = pre_processing(observe)

        action = np.zeros([action_size])
        rand_num = random.randint(0,2)
        action[rand_num] = 1

        reward=0
        done=False
        for i in range(skip_size):
            observe, reward1, done1 = agent.frame_step(action)
            if reward1 == -1:
                reward = -1
            elif reward1 == 1:
                reward = 1
            if done1 == True:
                done = True

        next_state = pre_processing(observe)

        #hidden_stat = sess.run(model.rnn_state, feed_dict={image : state, model.batch_size:1,model.trainLength:1})

        score += reward


        replay_memory.append([state, reward, action, next_state, done])

        state = next_state

        if len(replay_memory) > memory_size:
            replay_memory = np.delete(replay_memory,(0),axis=0)


    else:
        states = 'Training'

        if global_step == train_start:
            hidden_stat = (np.zeros([1, h_size]), np.zeros([1, h_size]))
            #hidden_stat = np.zeros([1, h_size])

        action = np.zeros([action_size])
        if random.random() < epsilon:
            action[random.randint(0,2)] = 1

            Image_in = np.reshape(state, (1, 80, 80, 1))
            hidden_stat_next = sess.run(model.rnn_state, feed_dict={image: Image_in, model.state_in: hidden_stat,
                                                                    model.batch_size: 1, model.trainLength: 1})


        else:
            Image_in = np.reshape(state,(1,80,80,1))
            Q = model.output.eval(feed_dict = {image: Image_in, model.rnn_state: hidden_stat,
                                               model.batch_size: 1, model.trainLength: 1})

            action = np.zeros([action_size])
            action[np.argmax(Q)] = 1

            hidden_stat_next = sess.run(model.rnn_state, feed_dict={image: Image_in, model.state_in: hidden_stat,
                                                                    model.batch_size: 1, model.trainLength: 1})



        reward=0
        done=False

        for i in range(skip_size):
            observe, reward1, done1 = agent.frame_step(action)
            if reward1 == -1:
                reward = -1
            elif reward1 == 1:
                reward = 1

            if done1 == True:
                done = True

        score += reward

        next_state = pre_processing(observe)

        replay_memory.append([state, reward, action, next_state, done])

        state = next_state
        hidden_stat = hidden_stat_next

        #print(len(replay_memory))

        if len(replay_memory) >= memory_size:
            replay_memory = np.delete(replay_memory, 0, axis=0)
            replay_memory = replay_memory.tolist()
            #print('after',len(replay_memory))

        if global_step % update_target_rate == 0:#######################################
            target_update()

        ##training time
        if global_step% train_step ==0:

            #state_train = np.zeros([batch_size, h_size])
            state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))
            mini_batch = get_sample(batch_size, train_step)  # (32*8 ,5)
            #print('mini_batch_size',np.shape(mini_batch))

            Image_batch = [batch[0] for batch in mini_batch]
            #print('Image_batch_size:',np.shape(Image_batch)) ##########################################################################################
            reward_batch = [batch[1] for batch in mini_batch]
            action_batch = [batch[2] for batch in mini_batch]
            next_Image_batch = [batch[3] for batch in mini_batch]
            done_batch = [batch[4] for batch in mini_batch]

            y_batch = []

            Q_batch = target_model.output.eval(
                feed_dict={image: next_Image_batch, target_model.rnn_state: state_train, target_model.batch_size: batch_size,
                           target_model.trainLength: train_step})


            for i in range(len(mini_batch)):
                if done_batch[i]==True:
                  y_batch.append(reward_batch[i])
                else:
                   y_batch.append(reward_batch[i]+decay_rate*np.max(Q_batch[i]))

            #print('y_batch_size',np.shape(y_batch))

            updateModel.run(feed_dict = {image:Image_batch, action_target:action_batch, y_prediction:y_batch,
                                     model.trainLength:train_step, model.batch_size:batch_size})

        if epsilon > epsilon_end:
            epsilon -=(epsilon_start-epsilon_end)/epsilon_step

    global_step += 1


    if done == True:
        print('episode:', episode,'    global_step:', global_step,  '    state:', states, '    epsilon:', epsilon, '    score:', score,  '    memory length',len(replay_memory))

        plot_x.append(episode)
        plot_y.append(score)


        episode += 1
        score = 0
        hidden_stat = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        #hidden_stat = np.zeros([1, h_size])


    if len(plot_x) % plot_episode == 0 and len(plot_x) != 0:
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Deep Q Learning')
        plt.grid(True)

        plt.plot(np.average(plot_x), np.average(plot_y), hold=True, marker='*', ms=5)
        plt.draw()
        plt.pause(0.000001)

        if np.average(plot_y)>3:
            terminal=0

        plot_x = []
        plot_y = []

    if global_step%40000 == 0:
        saver.save(sess,'gihoon/'+'pong_rnn')



    if episode == EPISODE:
        terminal = 0

    if terminal ==0:
        plt.savefig('DQNplot_RNN'+skip_size+'.png')





test_score = 0
test_episode = 0
test_step=0
end=1
test_x = []
test_y = []

plt.figure(2)

while(end):
    states = 'Testing'
    epsilon = 0

    Image_in = np.reshape(state, (1, 80, 80, 4))
    Q = model.output.eval(feed_dict={image: Image_in, model.state_in: hidden_stat,
                                     model.batch_size: 1, model.trainLength: 1})
    action = np.zeros([action_size])
    action[np.argmax(Q)] = 1

    reward = 0
    done = False

    for i in range(skip_size):
        observe, reward1, done1 = agent.frame_step(action)
        if reward1 == -1:
            reward = -1
        elif reward1 == 1:
            reward = 1

        if done1 == True:
            done = True

    test_score+=reward

    next_state = pre_processing(observe)

    state = next_state

    test_step += 1


    if done == True:
        print('episode:', test_episode, '    state:', states, '    score:', test_score)

        test_x.append(test_episode)
        test_y.append(test_score)

        test_episode += 1
        test_score = 0

        hidden_stat = (np.zeros([1, h_size]), np.zeros([1, h_size]))



    if len(test_x)%20 ==0 and len(test_x) != 0:
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Deep Q Learning')
        plt.grid(True)

        plt.plot(np.average(test_x), np.average(test_y), hold=True, marker='*', ms=5)
        plt.draw()
        plt.pause(0.000001)

        test_x=[]
        test_y=[]


    if test_episode==200:
        end=0

        plt.savefig('DQNplot_test_RNN'+skip_size+'.png')


















