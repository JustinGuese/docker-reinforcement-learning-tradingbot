import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C,DDPG, DQN, HerReplayBuffer, PPO, SAC, TD3

from os import environ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

from datetime import datetime

from ta import add_all_ta_features

import json


EPISODES = 1000000

# load json file bestvalstore.json and read it into python dictionary
with open('persistent/bestvalstore.json') as f:
    CONFIG = json.load(f)

df = yf.download("ETH-USD", period="730d",interval="1h")
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

# print(df.shape)
df.head()

# VARIABLES

WINDOW = 20
TRAINTESTSPLIT = .95
TRAINLOC = int(TRAINTESTSPLIT * len(df))

def saveConfig(bestreward,bestprofit,episodes):
    tmp = {
        "bestreward" : bestreward,
        "bestprofit" : bestprofit,
        "bestreward_date" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_episodes" : episodes
    }
    with open('persistent/bestvalstore.json') as f:
        json.dumps(tmp, f, indent=4)

def my_processed_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:,"Close"].to_numpy()[start:end]
    signal_features = env.df.loc[:, :].to_numpy()[start:end]# ['Close', 'Momentumbla']
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    def __init__(self, df, window_size, frame_bound):
        StocksEnv.__init__(self, df, window_size, frame_bound)
        StocksEnv.trade_fee_bid_percent = 0.00075 # binance fees https://www.binance.com/en/fee/schedule
        StocksEnv.trade_fee_ask_percent = 0.00075

    _process_data = my_processed_data
    
# make model

env = MyCustomEnv(df=df, window_size=WINDOW, frame_bound=(WINDOW,TRAINLOC)) # training data
env_training = lambda: env # gym.make("stocks-v0", df=df, frame_bound=(5, 200), window_size=5)
env = DummyVecEnv([env_training])
testenv = MyCustomEnv(df=df, window_size=WINDOW, frame_bound=(TRAINLOC,len(df))) # test data # gym.make("stocks-v0", df=df, frame_bound=(200, 300), window_size=5)


# create model
model = A2C("MlpPolicy", env, learning_rate= 0.01,  verbose=2)
# train
model.learn(total_timesteps=EPISODES)

# test data
# show it the ones it hasn't seen yet
env = MyCustomEnv(df=df, window_size=WINDOW, frame_bound=(TRAINLOC,len(df))) # test data # gym.make("stocks-v0", df=df, frame_bound=(200, 300), window_size=5)
obs = env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    if done:
        print(info)
        if info.get("total_reward") > CONFIG.get("bestreward"):
            print("improvement! saving now. best reward: %.2f, previous reward: %.2f" % (info.get("total_reward"), CONFIG.get("bestreward")))
            saveConfig(info.get("total_reward"),info.get("total_profit"),int(CONFIG.get("total_episodes"))+EPISODES)
            # save model
            model.save("persistent/a2cmlp.hf5")
        else:
            print("no improvement :( best reward: %.2f, previous reward: %.2f" % (info.get("total_reward"), CONFIG.get("bestreward")))
        break

# if profit < 1, then we made negative