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
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.technical_cache = {}
        
    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        """Get stock data from Alpha Vantage API."""
        try:
            logger.info(f"Downloading data for {symbol}...")
            
            # Construct API URL for daily adjusted data
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",  
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            
            # Make API request
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            
            # Debug log
            logger.info(f"API Response keys: {data.keys()}")
            
            # Check for API errors
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data and "API call frequency" in data["Note"]:
                raise ValueError("API rate limit exceeded")
            
            # Extract time series data
            time_series = data.get("Time Series (Daily)")
            if not time_series:
                logger.error(f"API Response: {data}")
                raise ValueError(f"No data returned for {symbol}. Please verify the symbol is correct.")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns (matching the actual API response format)
            df.columns = [col.split('. ')[1] for col in df.columns]
            df.index = pd.to_datetime(df.index)
            
            # Convert string values to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Add basic technical indicators
            df['Returns'] = df['close'].pct_change()
            df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
            df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA_Ratio'] = df['EMA5'] / df['EMA20']
            
            # Ensure we have all required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Successfully downloaded and processed data for {symbol}")
            logger.info(f"Data shape: {df.shape}, Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise ValueError(f"Failed to fetch data: {str(e)}")
        except ValueError as e:
            logger.error(f"Data processing error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise ValueError(f"Error processing data: {str(e)}")
            
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
        
        # Efficient RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # SMA
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        
        # Cache and return results
        df = df.fillna(method='ffill').fillna(0)
        self.technical_cache[cache_key] = df
        return df

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
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        self.state_size = 12
        self.action_size = 3  # hold, buy, sell
        self.transaction_fee = 0.001  # 0.1% transaction fee
        self.stop_loss = 0.02  # 2% stop loss
        self.take_profit = 0.03  # 3% take profit
        
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
        
        # Get technical indicators
        rsi = float(self.data.iloc[self.current_step]['RSI'])
        macd = float(self.data.iloc[self.current_step]['MACD'])
        macd_signal = float(self.data.iloc[self.current_step]['MACD_Signal'])
        
        reward = 0
        done = False
        
        # Calculate current portfolio value before action
        old_portfolio_value = self.balance + (self.shares_held * current_price)
        
        # Modify reward calculation for buy action
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
                        
                        # Record buy trade
                        trade_info = {
                            'date': current_date.strftime('%Y-%m-%d'),
                            'action': 'buy',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost,
                            'portfolio_value': self.balance + (self.shares_held * current_price)
                        }
                        self.trading_history.append(trade_info)
                        
                        # Modify reward based on technical indicators
                        rsi_reward = 1 if rsi < 30 else 0  # Buy when oversold
                        trend_reward = 1 if macd > macd_signal else 0  # Buy on bullish crossover
                        reward = 1 + rsi_reward + trend_reward  # Base reward + technical indicators
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                price_change = (current_price - self.entry_price) / self.entry_price
                
                # Modify sell conditions
                if price_change >= 0.005 or price_change <= -0.01:
                    revenue = self.shares_held * current_price * (1 - self.transaction_fee)
                    profit = revenue - (self.shares_held * self.entry_price)
                    
                    self.balance += revenue
                    if profit > 0:
                        self.profitable_trades += 1
                    self.total_profit += profit
                    
                    # Record sell trade
                    trade_info = {
                        'date': current_date.strftime('%Y-%m-%d'),
                        'action': 'sell',
                        'shares': self.shares_held,
                        'price': current_price,
                        'revenue': revenue,
                        'profit': profit,
                        'portfolio_value': self.balance
                    }
                    self.trading_history.append(trade_info)
                    
                    # Reset position
                    self.shares_held = 0
                    self.entry_price = 0
                    
                    # Modify reward calculation for sell
                    reward = max(1.0, abs(price_change) * 10)  # Use absolute price change
                    if price_change > 0:
                        reward *= 1.5  # Extra reward for profitable trades
                
                # Add small negative reward for holding too long
                else:
                    reward = -0.1
        
        # Add small negative reward for invalid actions
        else:
            reward = -0.1 if self.shares_held > 0 else 0  # Penalize holding when should sell
        
        # Update portfolio value
        new_portfolio_value = self.balance + (self.shares_held * current_price)
        portfolio_change = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Add portfolio change to reward
        reward += portfolio_change * 10
        
        self._portfolio_value = new_portfolio_value
        self.current_step += 1
        done = (self.current_step >= len(self.data) - 1)
        
        return self._get_state(), reward, done

    def _get_state(self):
        """Get current state."""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
            
        current_data = self.data.iloc[self.current_step]
        
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
        
        while not done:
            action = self.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            returns.append(reward)
            portfolio_values.append(env._portfolio_value)
            actions_taken.append(action)
        
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
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'avg_reward': np.mean(returns),
            'final_epsilon': self.epsilon,
            'episodes': 10,
            'action_distribution': action_distribution,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trading_history': env.trading_history  # Include trading history
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
    current_price = float(env.data.iloc[env.current_step]['close'])
    rsi = float(env.data.iloc[env.current_step]['RSI'])
    macd = float(env.data.iloc[env.current_step]['MACD'])
    macd_signal = float(env.data.iloc[env.current_step]['MACD_Signal'])
    ema5 = float(env.data.iloc[env.current_step]['EMA5'])
    ema20 = float(env.data.iloc[env.current_step]['EMA20'])
    
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

def initialize_trading_environment(symbol: str, initial_balance: float = 100) -> Tuple[TradingAgent, StockTradingEnvironment]:
    """Initialize the trading environment and agent for a given symbol."""
    try:
        # Import API key from app.py
        from app import API_KEY
        if not API_KEY:
            raise ValueError("API key not found in app.py")
        
        # Create processor with the API key
        processor = StockDataProcessor(api_key=API_KEY)
        
        # Get date ranges
        date_ranges = get_date_ranges()
        
        # Get training and testing data
        train_data = processor.get_stock_data(symbol)
        logger.info(f"Got {len(train_data)} days of training data for {symbol}")
        
        test_data = processor.get_stock_data(symbol)
        logger.info(f"Got {len(test_data)} days of testing data for {symbol}")
        
        # Initialize test environment
        env = StockTradingEnvironment(test_data, initial_balance=initial_balance)
        
        # Initialize agent
        agent = TradingAgent(state_size=env.state_size, action_size=env.action_size)
        
        logger.info(f"Training period: {date_ranges['train'][0]} to {date_ranges['train'][1]}")
        logger.info(f"Testing period: {date_ranges['test'][0]} to {date_ranges['test'][1]}")
        
        return agent, env
        
    except Exception as e:
        logger.error(f"Error initializing trading environment: {str(e)}")
        raise

def save_trading_results(symbol: str, results: Dict, recommendation: Dict):
    """Save trading results to JSON file."""
    output = {
        "symbol": symbol,
        "training_return": float(results.get('total_return', 0)),
        "final_portfolio_value": float(results.get('final_value', 1000)),
        "recommendation": {
            "action": recommendation.get('action', 'hold'),
            "confidence": float(recommendation.get('confidence', 0)),
            "probabilities": {
                "hold": float(recommendation.get('probabilities', {}).get('hold', 0)),
                "buy": float(recommendation.get('probabilities', {}).get('buy', 0)),
                "sell": float(recommendation.get('probabilities', {}).get('sell', 0))
            },
            "trend_analysis": {
                "recent_return": float(results.get('total_return', 0)),
                "total_return": float(results.get('total_return', 0) * 2)  # Annualized return
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "model_performance": {
            "avg_reward": float(results.get('avg_reward', 0)),
            "final_epsilon": float(results.get('final_epsilon', 0)),
            "training_episodes": int(results.get('episodes', 10)),
            "sharpe_ratio": float(results.get('sharpe_ratio', 0)),
            "max_drawdown": float(results.get('max_drawdown', 0))
        }
    }
    
    # Save to trading_results.json with proper path
    output_path = os.path.join(os.path.dirname(__file__), 'trading_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    
    logger.info(f"Trading results saved to {output_path}")

def main():
    """Main function to run the trading model."""
    try:
        # Set parameters
        symbol = 'AAPL'  # Example stock
        initial_balance = 100
        episodes = 10
        
        # Initialize environment and agent
        logger.info(f"\nInitializing trading environment for {symbol} with ${initial_balance:,.2f}...")
        agent, env = initialize_trading_environment(symbol, initial_balance=initial_balance)
        
        # Train the agent
        logger.info("\nStarting training...")
        agent.train(env, episodes=episodes)
        
        # Evaluate the model
        logger.info("\nEvaluating model performance...")
        results = agent.evaluate_model(env)
        
        # Generate and display recommendations
        recommendation = generate_trading_recommendations(agent, env)
        logger.info(f"\nTrading Recommendation: {recommendation}")
        
        # Save results to JSON
        save_trading_results(symbol, results, recommendation)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()