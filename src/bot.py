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

from pathlib import Path

import json

if environ.get('EPISODES'):
    EPISODES = int(environ.get('EPISODES'))
else:
    EPISODES = 1000000
    
MODEL_FILENAME = "persistent/a2cmlp.hf5"
VALSTORE_FILENAME = "persistent/bestvalstore.json"


def saveConfig(bestreward,bestprofit,episodes):
    tmp = {
        "bestreward" : bestreward,
        "bestprofit" : bestprofit,
        "bestreward_date" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_episodes" : episodes
    }
    with open(VALSTORE_FILENAME, "w") as f:
        json.dump(tmp, f, indent=4)

# load json file bestvalstore.json and read it into python dictionary
my_file = Path(VALSTORE_FILENAME)
if my_file.is_file():
    with open(VALSTORE_FILENAME, "r") as f:
        CONFIG = json.load(f)
else:
    saveConfig(0.,0.,0)
    


df = yf.download("ETH-USD", period="730d",interval="1h")
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

# print(df.shape)
df.head()

# VARIABLES

WINDOW = 20
TRAINTESTSPLIT = .95
TRAINLOC = int(TRAINTESTSPLIT * len(df))


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
    


env = MyCustomEnv(df=df, window_size=WINDOW, frame_bound=(WINDOW,TRAINLOC)) # training data
env_training = lambda: env # gym.make("stocks-v0", df=df, frame_bound=(5, 200), window_size=5)
env = DummyVecEnv([env_training])
testenv = MyCustomEnv(df=df, window_size=WINDOW, frame_bound=(TRAINLOC,len(df))) # test data # gym.make("stocks-v0", df=df, frame_bound=(200, 300), window_size=5)

# make model
# create model
# if model exists load it, otherwise create a new one
my_file = Path(MODEL_FILENAME)
if my_file.is_file():
    print("loading model")
    # file exists
    model = A2C.load(MODEL_FILENAME, env)
else:
    print("creating new model")
    model = A2C("MlpPolicy", env, learning_rate= 0.01,  verbose=0)

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
        if info.get("total_profit") > (CONFIG.get("bestprofit") + 0.001): # avoid really small changes
            print("improvement! saving now. best profit: %.2f, previous profit: %.2f" % (info.get("total_profit"), CONFIG.get("bestprofit")))
            saveConfig(info.get("total_reward"),info.get("total_profit"),int(CONFIG.get("total_episodes"))+EPISODES)
            # save model
            model.save(MODEL_FILENAME)
        else:
            print("no improvement :( best reward: %.2f, previous reward: %.2f" % (info.get("total_reward"), CONFIG.get("bestreward")))
        break

# if profit < 1, then we made negative