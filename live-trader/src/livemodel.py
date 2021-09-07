from gym_anytrading.envs import StocksEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from ta import add_all_ta_features
from os import environ
import yfinance as yf
import numpy as np
import requests
import json

MODEL_FILENAME = "persistent/a2cmlp.hf5"
VALSTORE_FILENAME = "persistent/bestvalstore.json"

STOCK = "ETH-USD"
USER = "elron@byom.de"
PW = "1234"

# get current data
df = yf.download(STOCK, period="2d",interval="1h")
print(df.tail())
# df = df.reset_index()
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

WINDOW = 20

df = df.iloc[-WINDOW-2:len(df)]
## prepare data

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
    
env = MyCustomEnv(df=df, window_size=WINDOW, frame_bound=(WINDOW,len(df))) # training data
env_training = lambda: env # gym.make("stocks-v0", df=df, frame_bound=(5, 200), window_size=5)
env = DummyVecEnv([env_training])
testenv = MyCustomEnv(df=df, window_size=WINDOW, frame_bound=(len(df),len(df))) # test data # gym.make("stocks-v0", df=df, frame_bound=(200, 300), window_size=5)

# LOAD MODEL
model = A2C.load("a2cmlp_bestprofit.hf5", env)

# PREDICT
preds = []

obs = env.reset()
action, _states = model.predict(obs)
obs, rewards, dones, info = env.step(action)
preds.append(action[0])
action, _states = model.predict(obs)
obs, rewards, dones, info = env.step(action)
preds.append(action[0])

action = np.median(preds)

actiondict = {
    0 : "Sell",
    0.5 : "Hold",
    1 : "Buy"
}

print("I PREDICT: %s" % actiondict[action], preds)

## functionality to communicate with fake backend
if action != 0.5:
    with requests.Session() as s:
            payload={'email': USER,
                    'password': PW}
            p = s.post('%s/login' % environ["ACCOUNT_URL"], data=payload)
            def buy(amount):
                # first get current holdings
                payload={'stock': STOCK,
                    'amount': amount}
                print("i am going to buy %d %s, because the current price is %2.f USD" % (amount, STOCK, crntPrice))
                p = s.post('%s/api/buyStock' % environ["ACCOUNT_URL"], json=payload)
                print(p.text)
                if json.loads(p.text)["code"] != 200:
                    print("something went wrong :(")
                
            def sell(owned):
                print("tryna sell %d %s" % (owned, STOCK))
                payload={'stock': STOCK}
                p = s.post('%s/api/sellStock' % environ["ACCOUNT_URL"], json=payload)
                print(p.text)
                if json.loads(p.text)["code"] != 200:
                    print("something went wrong :(")
            # should be logged in now
            # get current available eur
            resp = s.get('%s/api/getPortfolio' % environ["ACCOUNT_URL"]).json()
            euravailable = float(resp["EUR"]["amount"])
            stockAlreadyOwned = 0
            if resp.get(STOCK):
                stockAlreadyOwned = float(resp[STOCK]["amount"])
            print("i have %.2f euro to spend" % euravailable)
            
            crntPrice = df.iloc[-1]["Close"]
            amount = int(euravailable / crntPrice)
            
            if action == 1:
                if amount == 0:
                    print(" cant buy bc im a poor fuck")
                else:
                    buy(amount)
            elif action == 0:
                if stockAlreadyOwned == 0:
                    print("i don't own any stock")
                else:
                    sell(stockAlreadyOwned)
            
            
