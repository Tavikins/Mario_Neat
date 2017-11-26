#NEAT/PEAS Super Mario Gym (OpenAI)

import numpy as np
import gym

env = gym.make('SuperMarioBros-1-1-Tiles-v0')
from random import random, randint
from peas.networks.rnn import NeuralNetwork
from peas.methods.neat import NEATPopulation, NEATGenotype

from keras.models import Sequential      # One layer after the other
from keras.layers import Embedding, LSTM, Conv2D, Dense, Flatten, Reshape

model = Sequential()
model.add(Conv2D(256, kernel_size=(4,4), activation='relu',input_shape=(13,16,2)))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu')) # input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
model.add(Flatten())       # Flatten input so as to have no problems with processing
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2**6,  activation='relu'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

class GymSuperMario(object):
    def __init__(self, max_steps = 8000, max_score = 2000):
        self.max_steps = max_steps
        self.max_score = max_score
        self.actions = []
        self.states = []
        self.rewards = []
        
    def _loop(self,network,test=False):
        steps = 0
        self.actions = []
        self.states = []
        self.rewards = []
        done = False
        reward = 0            
        state = np.array(env.reset()).flatten()
        state = np.stack((state,state),axis=1).flatten()
        print("s",state.shape)
        while steps < self.max_steps and not done:
            steps += 1
            state_new = state
            action_index = np.argmax(model.predict(network.feed(state)[:416].reshape(1,13,16,2)))
            print("a_i",action_index)
            if test and random() > 0.5:
                action_index = randint(0,63)
            action = np.array(np.unravel_index(action_index,(2,2,2,2,2,2)))
            print("a",action)
            state, reward, done, info = env.step(action)
            if reward ==40:
                reward = 0
            #print("S",state.shape)
            #print("sn",state_new.shape)
            #state = state.flatten()
            state = np.stack((state_new.reshape((13,16,2))[:,:,1],state),axis=1).flatten()
            self.actions.append(action_index)
            self.rewards.append(reward)
            self.states.append(state)
            
            #print("s",state)
            print("r",reward)
        
        print("Training...")
        inputs = np.zeros((len(self.states),) + (1,13,16,2))
        targets = np.zeros((len(self.actions),1,64))
        
        for i in range(len(self.states)):
            print(self.states[i].shape)
            state = self.states[i]
            action = self.actions[i]
            reward = self.rewards[i]
            print('Train:')
            print('action',action)
            print('reward',reward)
            inputs[i:i+1] = state.reshape((1,13,16,2))
            targets[i] = model.predict(network.feed(state)[:416].reshape(1,13,16,2)).flatten()
            targets[i,0,action] = reward
            
            model.train_on_batch(inputs[i],targets[i])
        print("Training Done")
        return (reward,steps)
        
        
    def evaluate(self,network):
        if not isinstance(network,NeuralNetwork):
            network = NeuralNetwork(network)
            
        score, steps = self._loop(network, test=True)
        score, steps = self._loop(network)
        
        return {'fitness':score/self.max_score,'steps':steps}
        
    def solve(self,network):
        if not isinstance(network,NeuralNetwork):
            network=NeuralNetwork(network)
            
        score, steps = self._loop(network)
            
        if score < self.max_score:
            print("Failed... Score: ",score/self.max_score," in ",steps," Steps")
            return 0
            
        return int(score > self.max_score)



genotype = lambda: NEATGenotype(inputs=416,
                                outputs=207,
                                weight_range=(-50.,50.),
                                types=['tanh'])

pop = NEATPopulation(genotype,popsize= 10)

eval_task = GymSuperMario(max_steps=100,max_score=2000)

pop.epoch(generations=100,evaluator = eval_task, solution = eval_task)
