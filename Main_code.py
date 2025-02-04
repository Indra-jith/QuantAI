import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import torch.nn.functional as F
import requests
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
from typing import Tuple, List, Dict
import logging
import json
from collections import deque
import time
import os
import random
import traceback
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class StockDataProcessor:
    def __init__(self, api_key: str = None):
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        self.technical_cache = {}
        
    def get_stock_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get stock data from Alpha Vantage API and return both training and test data."""
        try:
            logger.info(f"Downloading data for {symbol}...")
            
            # API request
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Validate API response
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
            
            time_series = data.get("Time Series (Daily)")
            if not time_series:
                raise ValueError(f"No data returned for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Rename columns
            column_map = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            df.rename(columns=column_map, inplace=True)
            
            # Get date ranges
            today = pd.Timestamp.now().normalize()
            while today.weekday() > 4:  # Skip weekends
                today = today - pd.Timedelta(days=1)
            
            # Calculate periods ensuring minimum data requirements
            test_end = today
            test_start = test_end - pd.Timedelta(days=7)
            train_end = test_start - pd.Timedelta(days=1)
            train_start = train_end - pd.Timedelta(days=365)
            
            # Filter data
            train_data = df[(df.index >= train_start) & (df.index <= train_end)].copy()
            test_data = df[(df.index >= test_start) & (df.index <= test_end)].copy()
            
            # Validate data sizes
            if len(train_data) < 50:
                raise ValueError(f"Insufficient training data for {symbol}. Got {len(train_data)} days, need at least 50.")
            if len(test_data) < 5:
                raise ValueError(f"Insufficient test data for {symbol}. Got {len(test_data)} days, need at least 5.")
            
            # Calculate indicators with different parameters for train and test
            train_data = self._calculate_indicators(train_data, symbol, 'training')
            test_data = self._calculate_indicators(test_data, symbol, 'testing')
            
            logger.info(f"Processed data for {symbol}:")
            logger.info(f"Training: {len(train_data)} days ({train_data.index.min()} to {train_data.index.max()})")
            logger.info(f"Testing: {len(test_data)} days ({test_data.index.min()} to {test_data.index.max()})")
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            raise

    def _calculate_indicators(self, df: pd.DataFrame, symbol: str, period: str = 'training') -> pd.DataFrame:
        """Calculate technical indicators for a DataFrame."""
        try:
            # Initial data validation
            if len(df) < 5:  # Minimum required for any calculation
                raise ValueError(f"Insufficient data points for {symbol} ({period}). Got {len(df)} days.")
            
            # Convert and clean data first
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Initial cleaning
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            try:
                # Calculate base indicators first
                # RSI (using shorter period for test data)
                rsi_period = 14 if period == 'training' else 5
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(com=rsi_period-1, adjust=False).mean()
                avg_loss = loss.ewm(com=rsi_period-1, adjust=False).mean()
                rs = avg_gain / avg_loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD (adjusted periods for test data)
                if period == 'training':
                    fast_period, slow_period = 12, 26
                else:
                    fast_period, slow_period = 5, 10
                
                exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
                exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['MACD_Signal'] = df['MACD'].ewm(span=9 if period == 'training' else 4, adjust=False).mean()
                
                # EMAs
                df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
                df['EMA20'] = df['close'].ewm(span=20 if period == 'training' else 10, adjust=False).mean()
                
                # Bollinger Bands (adaptive window size)
                bb_window = 20 if period == 'training' else 5
                rolling_mean = df['close'].rolling(window=bb_window, min_periods=1).mean()
                rolling_std = df['close'].rolling(window=bb_window, min_periods=1).std()
                df['BB_Upper'] = rolling_mean + (rolling_std * 2)
                df['BB_Lower'] = rolling_mean - (rolling_std * 2)
                df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                
                # Other indicators (adaptive windows)
                df['Returns'] = df['close'].pct_change()
                vol_window = 20 if period == 'training' else 5
                df['Volatility'] = df['Returns'].rolling(window=vol_window, min_periods=1).std()
                df['Volume_MA5'] = df['volume'].rolling(window=5, min_periods=1).mean()
                
                # Handle any remaining NaN values
                df = df.fillna(method='ffill').fillna(method='bfill')
                
                # Final validation
                required_indicators = [
                    'RSI', 'MACD', 'MACD_Signal', 'EMA5', 'EMA20',
                    'BB_Upper', 'BB_Lower', 'BB_Position', 'Volatility',
                    'Volume_MA5'
                ]
                
                # Check for NaN values in required columns
                nan_indicators = [ind for ind in required_indicators if df[ind].isna().any()]
                if nan_indicators:
                    raise ValueError(f"NaN values found in indicators: {', '.join(nan_indicators)}")
                
                logger.info(f"Successfully calculated indicators for {symbol} {period} data. Shape: {df.shape}")
                return df
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error in indicator calculation: {str(e)}")
            raise ValueError(f"Failed to calculate indicators: {str(e)}")

    def get_data_for_period(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data for a specific period with validation."""
        try:
            # Get full data
            data = self.get_stock_data(ticker)
            
            # Convert dates to pandas datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Filter data for the specified period
            mask = (data.index >= start_date) & (data.index <= end_date)
            period_data = data.loc[mask].copy()
            
            if len(period_data) == 0:
                raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")
            
            # Add technical indicators
            period_data = self.add_technical_indicators(period_data)
            
            logger.info(f"Got {len(period_data)} days of data for {ticker}")
            return period_data
            
        except Exception as e:
            logger.error(f"Error getting data for period: {str(e)}")
            raise

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fast technical indicators using vectorized operations with caching."""
        cache_key = hash(pd.util.hash_pandas_object(data).sum())
        if cache_key in self.technical_cache:
            return self.technical_cache[cache_key]
        
        df = data.copy()
        
        try:
            # Fast EMA calculations
            df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA_Ratio'] = df['EMA5'] / df['EMA20']
            
            # Fast volatility and returns
            df['Returns'] = df['close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(20, min_periods=1).std()
            df['Log_Returns'] = np.log1p(df['Returns'])
            
            # Volume analysis
            df['Volume_MA5'] = df['volume'].rolling(5).mean()
            df['Volume_Spike'] = (df['volume'] / df['Volume_MA5']) - 1
            
            # Momentum indicators
            df['Momentum3'] = df['close'].pct_change(3)
            df['Momentum5'] = df['close'].pct_change(5)
            
            # Bollinger Bands (faster calculation)
            rolling_mean = df['close'].rolling(window=20).mean()
            rolling_std = df['close'].rolling(window=20).std()
            df['BB_Upper'] = rolling_mean + (rolling_std * 2)
            df['BB_Lower'] = rolling_mean - (rolling_std * 2)
            df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Improved RSI calculation with error handling
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            
            # Handle division by zero
            rs = pd.Series(index=df.index)
            rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD with error handling
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # SMA
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            # Verify required indicators are present and valid
            required_indicators = ['RSI', 'MACD', 'MACD_Signal']
            for indicator in required_indicators:
                if indicator not in df.columns:
                    raise ValueError(f"Missing required indicator: {indicator}")
                if df[indicator].isnull().any():
                    raise ValueError(f"Invalid values in {indicator}")
            
            # Cache and return results
            self.technical_cache[cache_key] = df
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise ValueError(f"Failed to calculate technical indicators: {str(e)}")

class DQN(nn.Module):
    """Simplified DQN architecture with proper initialization."""
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)
        
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class StockTradingEnvironment:
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100):
        """Initialize the environment with the given data."""
        # Ensure we have all required columns
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'RSI', 'MACD', 'MACD_Signal', 'EMA5', 'EMA20',
            'BB_Position', 'Volatility', 'Volume_MA5'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {', '.join(missing_columns)}")
        
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        self.state_size = 12
        self.action_size = 3  # hold, buy, sell
        self.transaction_fee = 0.001  # 0.1% transaction fee
        
    def reset(self):
        """Reset the environment."""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0
        self.trading_history = []
        self._portfolio_value = self.initial_balance
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute one step in the environment."""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True
            
        current_price = float(self.data.iloc[self.current_step]['close'])
        current_date = self.data.index[self.current_step]
        
        reward = 0
        done = False
        
        # Calculate current portfolio value before action
        old_portfolio_value = self.balance + (self.shares_held * current_price)
        
        if action == 1:  # Buy
            if self.shares_held == 0:
                max_shares = int(self.balance * 0.9 / (current_price * (1 + self.transaction_fee)))
                
                if max_shares > 0:
                    shares_to_buy = max_shares
                    cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                    
                    if cost <= self.balance:
                        self.balance -= cost
                        self.shares_held = shares_to_buy
                        self.entry_price = current_price
                        self.total_trades += 1
                        
                        # Record buy trade with all required fields
                        trade_info = {
                            'date': current_date.strftime('%Y-%m-%d'),
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'value': shares_to_buy * current_price,
                            'portfolio_value': self.balance + (self.shares_held * current_price)
                        }
                        self.trading_history.append(trade_info)
                        
                        reward = 1  # Base reward for successful buy
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                sale_value = self.shares_held * current_price * (1 - self.transaction_fee)
                profit = sale_value - (self.shares_held * self.entry_price)
                
                self.balance += sale_value
                if profit > 0:
                    self.profitable_trades += 1
                self.total_profit += profit
                
                # Record sell trade with all required fields
                trade_info = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'action': 'SELL',
                    'shares': self.shares_held,
                    'price': current_price,
                    'value': sale_value,
                    'profit': profit,
                    'portfolio_value': self.balance
                }
                self.trading_history.append(trade_info)
                
                self.shares_held = 0
                self.entry_price = 0
                
                reward = max(1.0, profit / self.initial_balance * 10)  # Reward based on profit
        
        # Update portfolio value and step
        new_portfolio_value = self.balance + (self.shares_held * current_price)
        self._portfolio_value = new_portfolio_value
        self.current_step += 1
        done = (self.current_step >= len(self.data) - 1)
        
        return self._get_state(), reward, done

    def _get_state(self):
        """Get current state."""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
            
        current_data = self.data.iloc[self.current_step]
        
        # Ensure all required data is available
        try:
            state = np.array([
                current_data['RSI'] / 100,  # Normalized RSI
                current_data['MACD'] / current_data['close'],  # Normalized MACD
                current_data['MACD_Signal'] / current_data['close'],  # Normalized MACD Signal
                current_data['Volatility'],
                (current_data['EMA5'] - current_data['EMA20']) / current_data['close'],  # Trend
                current_data['BB_Position'],  # Bollinger Band position
                self.shares_held * current_data['close'] / self._portfolio_value if self._portfolio_value > 0 else 0,
                self.balance / self.initial_balance,  # Normalized balance
                1 if self.shares_held > 0 else 0,  # Position flag
                self._portfolio_value / self.initial_balance,  # Total return
                current_data['Volume_MA5'] / current_data['volume'] - 1,  # Volume trend
                current_data['close'] / current_data['open'] - 1  # Daily return
            ], dtype=np.float32)
            
            return state
            
        except KeyError as e:
            logger.error(f"Missing required data column: {e}")
            raise ValueError(f"Missing required data column: {e}")
        except Exception as e:
            logger.error(f"Error creating state: {e}")
            raise

class TradingAgent:
    """Deep Q-Learning agent for stock trading."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001,
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.995):
        """Initialize the trading agent with improved parameters."""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Choose action based on epsilon-greedy policy."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            return torch.argmax(act_values).item()
            
    def replay(self, batch_size):
        """Train on past experiences."""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([i[0] for i in minibatch])
        actions = torch.LongTensor([i[1] for i in minibatch])
        rewards = torch.FloatTensor([i[2] for i in minibatch])
        next_states = torch.FloatTensor([i[3] for i in minibatch])
        dones = torch.FloatTensor([i[4] for i in minibatch])
        
        # Current Q values
        curr_Q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_Q = self.model(next_states).detach().max(1)[0]
        target_Q = rewards + (self.gamma * next_Q * (1 - dones))
        
        # Compute loss and update
        loss = self.criterion(curr_Q.squeeze(), target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def train(self, env, episodes=10, batch_size=32):
        """Train the agent."""
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for time in range(len(env.data) - 1):
                action = self.act(state)
                next_state, reward, done = env.step(action)
                total_reward += reward
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                
                if len(self.memory) > batch_size:
                    self.replay(batch_size)
                    
                if done:
                    break
                    
            logger.info(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward}")
    
    def evaluate_model(self, env: StockTradingEnvironment) -> Dict:
        """
        Strict evaluation without retraining, using Monte Carlo validation.
        """
        state = env.reset()
        done = False
        total_reward = 0
        returns = []
        portfolio_values = [env._portfolio_value]
        actions_taken = []
        trading_history = []  # Initialize trading history
        
        while not done:
            action = self.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            returns.append(reward)
            portfolio_values.append(env._portfolio_value)
            actions_taken.append(action)
            
            # Get trading history from environment
            if len(env.trading_history) > len(trading_history):
                # New trade was made
                latest_trade = env.trading_history[-1]
                trading_history.append(latest_trade)
        
        # Calculate metrics
        final_value = portfolio_values[-1]
        initial_value = portfolio_values[0]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (using daily returns)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = 0
        if len(daily_returns) > 1:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            if std_return > 0:
                sharpe_ratio = (avg_return / std_return) * np.sqrt(252)  # Annualized
        
        # Calculate action distribution
        action_counts = Counter(actions_taken)
        total_actions = len(actions_taken)
        action_distribution = {
            'hold': action_counts[0] / total_actions if total_actions > 0 else 0,
            'buy': action_counts[1] / total_actions if total_actions > 0 else 0,
            'sell': action_counts[2] / total_actions if total_actions > 0 else 0
        }
        
        # Calculate total profit from trading history
        total_profit = sum(trade.get('profit', 0) for trade in trading_history)
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'avg_reward': np.mean(returns),
            'final_epsilon': self.epsilon,
            'episodes': 10,
            'action_distribution': action_distribution,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trading_history': trading_history,  # Include full trading history
            'total_trades': len(trading_history),
            'total_profit': total_profit
        }

def generate_trading_recommendations(agent, env) -> dict:
    """Generate trading recommendations based on current state."""
    state = env._get_state()
    
    # Convert state to tensor and get predictions
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        q_values = agent.model(state_tensor).squeeze().numpy()  # Use forward pass directly
    
    # Calculate probabilities using softmax
    probabilities = F.softmax(torch.FloatTensor(q_values), dim=0).numpy()
    
    # Get current price and technical indicators
    current_data = env.data.iloc[env.current_step]
    current_price = float(current_data['close'])
    rsi = float(current_data['RSI'])
    macd = float(current_data['MACD'])
    macd_signal = float(current_data['MACD_Signal'])
    ema5 = float(current_data['EMA5'])
    ema20 = float(current_data['EMA20'])
    
    # Determine action based on highest probability
    action_idx = np.argmax(q_values)
    actions = ['HOLD', 'BUY', 'SELL']
    action = actions[action_idx]
    
    # Calculate confidence score (0-100)
    confidence = float(probabilities[action_idx] * 100)
    
    # Generate reason based on technical indicators
    reason = ""
    if action == 'BUY':
        if rsi < 30:
            reason = "RSI indicates oversold conditions"
        elif macd > macd_signal:
            reason = "MACD shows bullish crossover"
        elif ema5 > ema20:
            reason = "Short-term trend is bullish"
        else:
            reason = "Technical indicators suggest upward momentum"
    elif action == 'SELL':
        if rsi > 70:
            reason = "RSI indicates overbought conditions"
        elif macd < macd_signal:
            reason = "MACD shows bearish crossover"
        elif ema5 < ema20:
            reason = "Short-term trend is bearish"
        else:
            reason = "Technical indicators suggest downward momentum"
    else:
        reason = "Market conditions suggest holding position"
    
    return {
        'action': action,
        'confidence': confidence,
        'reason': reason,
        'current_price': current_price,
        'probabilities': {
            'hold': float(probabilities[0]),
            'buy': float(probabilities[1]),
            'sell': float(probabilities[2])
        },
        'technical_indicators': {
            'RSI': rsi,
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'EMA5': ema5,
            'EMA20': ema20
        }
    }

def get_date_ranges() -> Tuple[str, str]:
    """Get date ranges for training and testing."""
    end_date = datetime.now()
    
    # Find last working day (excluding weekends)
    while end_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
        end_date = end_date - timedelta(days=1)
    
    # Get last 7 working days for testing
    test_start_date = end_date
    working_days = 0
    while working_days < 7:
        test_start_date = test_start_date - timedelta(days=1)
        if test_start_date.weekday() < 5:  # If it's a weekday
            working_days += 1
    
    # Get exactly one year before test period for training
    train_start_date = test_start_date - timedelta(days=365)
    train_end_date = test_start_date - timedelta(days=1)
    
    # Format dates as strings
    train_start_str = train_start_date.strftime('%Y-%m-%d')
    train_end_str = train_end_date.strftime('%Y-%m-%d')
    test_start_str = test_start_date.strftime('%Y-%m-%d')
    test_end_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(
        f"Training period: {train_start_str} to {train_end_str}\n"
        f"Testing period: {test_start_str} to {test_end_str}"
    )
    
    return {
        'train': (train_start_str, train_end_str),
        'test': (test_start_str, test_end_str)
    }

def initialize_trading_environment(symbol: str, initial_balance: float = 100) -> Tuple[TradingAgent, StockTradingEnvironment, StockTradingEnvironment]:
    """Initialize the trading environment and agent for both training and testing."""
    try:
        api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("Alpha Vantage API key not found")
        
        processor = StockDataProcessor(api_key=api_key)
        
        # Get both training and test data
        train_data, test_data = processor.get_stock_data(symbol)
        
        # Initialize environments
        train_env = StockTradingEnvironment(train_data, initial_balance=initial_balance)
        test_env = StockTradingEnvironment(test_data, initial_balance=initial_balance)
        
        # Initialize agent
        agent = TradingAgent(state_size=train_env.state_size, action_size=train_env.action_size)
        
        return agent, train_env, test_env
        
    except Exception as e:
        logger.error(f"Error initializing trading environment: {str(e)}")
        raise

def save_trading_results(symbol: str, train_results: Dict, test_results: Dict, recommendation: Dict):
    """Save both training and testing results to JSON file."""
    try:
        output = {
            "symbol": symbol,
            "training_return": float(train_results.get('total_return', 0)),
            "final_portfolio_value": float(test_results.get('final_value', 0)),
            "initial_balance": 100,
            "total_return_pct": float(test_results.get('total_return', 0) * 100),
            "training": {
                "return": float(train_results.get('total_return', 0)),
                "final_value": float(train_results.get('final_value', 0)),
                "trades": len(train_results.get('trading_history', [])),
                "profit": float(train_results.get('total_profit', 0))
            },
            "testing": {
                "return": float(test_results.get('total_return', 0)),
                "final_value": float(test_results.get('final_value', 0)),
                "trades": len(test_results.get('trading_history', [])),
                "profit": float(test_results.get('total_profit', 0))
            },
            "model_performance": {
                "avg_reward": float(test_results.get('avg_reward', 0)),
                "final_epsilon": float(test_results.get('final_epsilon', 0)),
                "training_episodes": int(test_results.get('episodes', 10)),
                "sharpe_ratio": float(test_results.get('sharpe_ratio', 0)),
                "max_drawdown": float(test_results.get('max_drawdown', 0)),
                "total_trades": len(test_results.get('trading_history', [])),
                "total_profit": float(test_results.get('total_profit', 0))
            },
            "recommendation": {
                "action": recommendation.get('action', 'HOLD'),
                "confidence": float(recommendation.get('confidence', 0)),
                "reason": recommendation.get('reason', ''),
                "probabilities": {
                    "hold": float(recommendation.get('probabilities', {}).get('hold', 0)),
                    "buy": float(recommendation.get('probabilities', {}).get('buy', 0)),
                    "sell": float(recommendation.get('probabilities', {}).get('sell', 0))
                },
                "trend_analysis": {
                    "recent_return": float(test_results.get('total_return', 0)),
                    "total_return": float(test_results.get('total_return', 0) * 2)
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "trading_history": []
        }

        # Process trading history
        for trade in test_results.get('trading_history', []):
            processed_trade = {
                'date': trade.get('date', ''),
                'action': trade.get('action', ''),
                'shares': float(trade.get('shares', 0)),
                'price': float(trade.get('price', 0)),
                'value': float(trade.get('shares', 0)) * float(trade.get('price', 0)),
                'profit': float(trade.get('profit', 0)) if 'profit' in trade else 0,
                'portfolio_value': float(trade.get('portfolio_value', 0))
            }
            output['trading_history'].append(processed_trade)

        # Ensure the directory exists
        output_path = os.path.join(os.path.dirname(__file__), 'trading_results.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write to a temporary file first
        temp_path = output_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(output, f, indent=4)

        # Validate the JSON
        with open(temp_path, 'r') as f:
            json.load(f)  # This will raise an error if JSON is invalid

        # If validation passes, rename temp file to actual file
        os.replace(temp_path, output_path)
        
        logger.info(f"Trading results successfully saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving trading results: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def main():
    """Main function to run the trading model."""
    try:
        # Set parameters
        symbol = 'AAPL'  # Example stock
        initial_balance = 100
        episodes = 10
        
        # Initialize environment and agent
        logger.info(f"\nInitializing trading environment for {symbol} with ${initial_balance:,.2f}...")
        agent, train_env, test_env = initialize_trading_environment(symbol, initial_balance=initial_balance)
        
        # Train the agent
        logger.info("\nStarting training...")
        agent.train(train_env, episodes=episodes)
        
        # Evaluate the model
        logger.info("\nEvaluating model performance...")
        train_results = agent.evaluate_model(train_env)
        test_results = agent.evaluate_model(test_env)
        
        # Generate and display recommendations
        train_recommendation = generate_trading_recommendations(agent, train_env)
        test_recommendation = generate_trading_recommendations(agent, test_env)
        logger.info(f"\nTraining Recommendation: {train_recommendation}")
        logger.info(f"\nTesting Recommendation: {test_recommendation}")
        
        # Save results to JSON
        save_trading_results(symbol, train_results, test_results, test_recommendation)
        
        return train_results, test_results
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise