from __future__ import annotations

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import defaultdict


class DataFeed:
    def pull(self):
        raise NotImplementedError
    def advance(self):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    def get_ticker_list(self):
        raise NotImplementedError

class DataFrameDataFeed(DataFeed):
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.timestep = 0
        self.df = df.copy()
        self.df = self.df.sort_values(['timestamp', 'tic'], ignore_index=True)
        self.df.index = self.df['timestamp'].factorize()[0]
        self.ticker_list = sorted(self.df.tic.unique())
    
    def pull(self):
        return  self.df.loc[self.timestep, :]
    
    def advance(self):
        if self.timestep >= len(self.df.index.unique()) - 1:
            return False
        self.timestep += 1
        return True

    def reset(self):
        self.timestep = 0

    def get_ticker_list(self):
        return self.ticker_list


class Account:
    def get_positions(self):
        raise NotImplementedError
    def get_cash(self):
        raise NotImplementedError
    def submit_order(self, qty, sym, side, price):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError

class SimulatedAccount(Account):
    def __init__(
        self,
        initial_cash: int,
        # todo: integrate flat+pct commission/fee params
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = defaultdict(int)

    def get_positions(self):
        return self.positions
    
    def get_cash(self):
        return self.cash

    def submit_order(self, qty, sym, side, price):
        if side =='buy':
            qty_to_buy = min(qty, int(self.cash / price))
            if qty_to_buy <= 0:
                return False
            self.positions[sym] += qty_to_buy
            self.cash -= qty_to_buy * price
        elif side == 'sell':
            qty_to_sell = min(self.positions[sym], qty)
            if qty_to_sell <= 0:
                return False
            self.positions[sym] -= qty_to_sell
            self.cash += qty_to_sell * price
        return True

    def reset(self):
        self.cash = self.initial_cash
        self.positions = defaultdict(int)

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        feed: DataFeed,
        account: Account,
        reward_scaling: float,
        tech_indicator_list: list[str],
        min_shares_per_order: int,
        max_shares_per_stock: int,
    ):
        self.feed = feed
        self.account = account
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        self.min_shares_per_order = min_shares_per_order
        self.max_shares_per_stock = max_shares_per_stock
        
        self.ticker_list = self.feed.get_ticker_list()
        self.stock_dim = len(self.ticker_list)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))

        self.timestep_df = self.feed.pull()
        self.initial_total_asset = self.get_account_value()
        self.total_asset = self.initial_total_asset
        self.stocks_cool_down = np.zeros(self.stock_dim)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.get_state()),)
        )
        self._seed()


    def get_positions(self):
        positions = [0] * self.stock_dim
        for sym, shares in self.account.get_positions().items():
            if sym in self.ticker_list:
                positions[self.ticker_list.index(sym)] = shares
        return positions

    def get_account_value(self):
        positions = self.get_positions()
        price = self.timestep_df.close.values
        return self.account.get_cash() + (positions * price).sum()

    def get_state(self):
        # todo: scaling that makes sense
        amount = np.array(self.total_asset * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        return np.hstack(
            (
                amount,
                self.timestep_df.close.values * scale,
                self.get_positions() * scale,
                self.stocks_cool_down,
                sum(
                        (
                            (self.timestep_df[tech].values * scale).tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
            )
        )

    def reset(self):
        self.feed.reset()
        self.account.reset()
        
        self.timestep_df = self.feed.pull()
        self.initial_total_asset = self.get_account_value()
        self.total_asset = self.initial_total_asset
        self.stocks_cool_down = np.zeros(self.stock_dim)

        return self.get_state()


    def step(self, actions):
        actions = (actions * self.max_shares_per_stock).astype(int)

        price = self.timestep_df['close'].values
        total_asset = self.get_account_value()

        self.stocks_cool_down += 1
        for index in np.where(actions < -self.min_shares_per_order)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                if self.account.submit_order(-actions[index], self.ticker_list[index], 'sell', price[index]):
                    self.stocks_cool_down[index] = 0
        for index in np.where(actions > self.min_shares_per_order)[0]:  # buy_index:
            if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)
                if self.account.submit_order(actions[index], self.ticker_list[index], 'buy', price[index]):
                    self.stocks_cool_down[index] = 0
                self.stocks_cool_down[index] = 0

        done = not self.feed.advance()
        self.timestep_df = self.feed.pull()
        state = self.get_state()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset

        return state, reward, done, {'total_asset': total_asset}


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
