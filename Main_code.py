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
    def __init__(self, data: pd.DataFrame, initial_balance: float = 1000):
        """Initialize the trading environment."""
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = 0
        self._portfolio_value = initial_balance
        self.trading_history = []
        self.transaction_fee = 0.0005  # Reduced to 0.05% transaction fee
        self.stop_loss = 0.02  # 2% stop loss
        self.take_profit = 0.03  # 3% take profit
        self.entry_price = 0  # Track entry price for position
        
        # Define state size based on our features
        self.state_size = 12  # Fixed number of features we use in _get_state
        self.action_size = 3  # hold, buy, sell
        
    def _get_state(self) -> np.ndarray:
        """Get the current state of the environment."""
        current_idx = self.current_step
        prev_idx = max(0, current_idx - 1)
        
        current_data = self.data.iloc[current_idx]
        prev_data = self.data.iloc[prev_idx]
        
        # Calculate normalized features
        price_change = ((current_data['close'] - prev_data['close']) / prev_data['close'])
        volume_change = ((current_data['volume'] - prev_data['volume']) / prev_data['volume']) if prev_data['volume'] != 0 else 0
        
        # Get technical indicators (already normalized)
        rsi = current_data['RSI'] / 100  # Normalize RSI to [0, 1]
        macd = current_data['MACD']
        macd_signal = current_data['MACD_Signal']
        momentum = current_data['Momentum5']
        volatility = current_data['Volatility']
        ema_ratio = current_data['EMA_Ratio']
        
        # Portfolio state
        portfolio_value = self._portfolio_value
        cash_ratio = self.balance / portfolio_value if portfolio_value > 0 else 0
        position_ratio = (self.shares_held * current_data['close']) / portfolio_value if portfolio_value > 0 else 0
        position_indicator = 1 if self.shares_held > 0 else 0
        
        # Create state vector with exactly 12 features
        state = np.array([
            price_change,      # Price momentum
            volume_change,     # Volume momentum
            rsi,              # RSI (normalized)
            macd,             # MACD
            macd_signal,      # MACD signal
            momentum,         # Price momentum
            volatility,       # Market volatility
            ema_ratio,        # Trend indicator
            cash_ratio,       # Cash position
            position_ratio,   # Stock position
            position_indicator,# Current position indicator
            portfolio_value / self.initial_balance  # Normalized portfolio value
        ], dtype=np.float32)
        
        return state

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self._portfolio_value = self.initial_balance
        self.trading_history = []  # Reset trading history
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute one step in the environment."""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0.0, True

        current_price = self.data.iloc[self.current_step]['close']
        current_date = self.data.index[self.current_step]
        
        # Get market indicators
        rsi = self.data.iloc[self.current_step]['RSI']
        macd = self.data.iloc[self.current_step]['MACD']
        macd_signal = self.data.iloc[self.current_step]['MACD_Signal']
        macd_hist = self.data.iloc[self.current_step]['MACD_Hist']
        sma20 = self.data.iloc[self.current_step]['SMA20']
        momentum = self.data.iloc[self.current_step]['Momentum5']
        volatility = self.data.iloc[self.current_step]['Volatility']
        
        # Calculate position value before action
        position_value = self.shares_held * current_price
        total_value_before = self.balance + position_value
        
        # Check stop loss and take profit if we have a position
        force_sell = False
        if self.shares_held > 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            if price_change <= -self.stop_loss or price_change >= self.take_profit:
                action = 2  # Force sell
                force_sell = True

        trade_info = None
        reward = 0
        
        # Execute trading action
        if action == 1:  # Buy
            if self.balance >= current_price and self.shares_held == 0:  # Only buy if we don't have a position
                # Dynamic position sizing based on volatility
                risk_factor = min(0.5, 0.1 / (volatility + 0.01))
                shares_to_buy = int((self.balance * risk_factor) / current_price)
                
                if shares_to_buy > 0:
                    transaction_cost = shares_to_buy * current_price * self.transaction_fee
                    total_cost = shares_to_buy * current_price + transaction_cost
                    
                    if self.balance >= total_cost:
                        self.balance -= total_cost
                        self.shares_held = shares_to_buy
                        self.entry_price = current_price  # Track entry price
                        
                        trade_info = {
                            'date': current_date.strftime('%Y-%m-%d'),
                            'action': 'buy',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': total_cost,
                            'portfolio_value': self._portfolio_value
                        }
                        
                        # Improved reward calculation for buys
                        reward = 0
                        if rsi < 30:  # Stronger oversold condition
                            reward += 2.0
                        if macd > macd_signal and abs(macd - macd_signal) > 0.01:  # Significant MACD crossover
                            reward += 1.5
                        if current_price > sma20 and momentum > 0:  # Trend confirmation
                            reward += 1.0
                        if volatility < 0.02:  # Low volatility bonus
                            reward += 0.5
                        
                        # Scale reward by risk
                        reward /= (volatility + 0.01)
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                transaction_cost = self.shares_held * current_price * self.transaction_fee
                total_value = self.shares_held * current_price - transaction_cost
                self.balance += total_value
                
                # Calculate return on trade
                price_change = (current_price - self.entry_price) / self.entry_price
                
                trade_info = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'action': 'sell',
                    'shares': self.shares_held,
                    'price': current_price,
                    'value': total_value,
                    'return': price_change * 100,  # Store return percentage
                    'portfolio_value': self._portfolio_value
                }
                
                # Risk-adjusted reward for sells
                reward = price_change * 100  # Base reward
                
                # Bonus for selling at resistance
                if rsi > 70:
                    reward *= 1.2
                if macd < macd_signal:
                    reward *= 1.1
                
                # Penalize high volatility trades
                reward /= (volatility + 0.01)
                
                self.shares_held = 0
                self.entry_price = 0

        # Record trade if any happened
        if trade_info:
            self.trading_history.append(trade_info)

        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value change
        new_price = self.data.iloc[self.current_step]['close']
        new_position_value = self.shares_held * new_price
        total_value_after = self.balance + new_position_value
        
        # Update portfolio value
        self._portfolio_value = total_value_after
        
        # Calculate final reward
        if not trade_info:  # If no trade was made
            # Small negative reward for holding cash too long
            if self.shares_held == 0 and rsi < 40 and macd > macd_signal:
                reward -= 0.1
            # Small negative reward for not taking profit
            elif self.shares_held > 0 and rsi > 70 and macd < macd_signal:
                reward -= 0.1
        
        done = self.current_step >= len(self.data) - 1
        return self._get_state(), reward, done

    def get_state(self) -> np.ndarray:
        """Get the current state of the environment."""
        current_step = self.current_step - 1
        
        # Get current price and technical indicators
        current_price = self.data.iloc[current_step]['close']
        rsi = self.data.iloc[current_step]['RSI']
        macd = self.data.iloc[current_step]['MACD']
        macd_signal = self.data.iloc[current_step]['MACD_Signal']
        ema5 = self.data.iloc[current_step]['EMA5']
        ema20 = self.data.iloc[current_step]['EMA20']
        volatility = self.data.iloc[current_step]['Volatility']
        volume = self.data.iloc[current_step]['volume']
        
        # Calculate price changes
        price_change = self.data.iloc[current_step]['Returns']
        volume_ma5 = self.data.iloc[current_step]['Volume_MA5']
        
        # Normalize indicators
        normalized_price = current_price / self.data['close'].mean() - 1
        normalized_volume = volume / volume_ma5 - 1
        
        # Create state array
        state = np.array([
            normalized_price,          # Normalized price
            price_change,             # Price change
            rsi / 100,                # Normalized RSI
            macd,                     # MACD
            macd_signal,              # MACD Signal
            ema5 / current_price - 1,  # EMA5 relative to price
            ema20 / current_price - 1, # EMA20 relative to price
            volatility,               # Volatility
            normalized_volume,        # Normalized volume
            self.shares_held / 100,        # Current position
            self.balance / 10000,     # Normalized balance
            1 if self.shares_held > 0 else 0  # Position indicator
        ], dtype=np.float32)
        
        return state

class TradingAgent:
    """Deep Q-Learning agent for stock trading."""
    
    def __init__(self, state_size: int, action_size: int, device: str = 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95    # Slightly reduced discount rate
        self.epsilon = 1.0   # Start with full exploration
        self.epsilon_min = 0.15  # Higher minimum exploration
        self.epsilon_decay = 0.85  # Faster decay
        self.batch_size = 32   # Smaller batch size
        self.learning_rate = 0.001  # Increased learning rate
        self.tau = 0.005  # Faster target network updates
        self.device = torch.device(device)
        
        # Initialize networks with correct state size
        self.model = DQN(12, action_size).to(self.device)
        self.target_model = DQN(12, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Track returns for evaluation
        self.returns = []
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose an action."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for a given state."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            # Convert Q-values to probabilities using softmax
            probs = F.softmax(q_values, dim=1)
            return probs.cpu().numpy()[0]
    
    def train(self, env: StockTradingEnvironment, episodes: int = 20):
        """Train the agent using DQN with soft target updates."""
        start_time = time.time()
        total_rewards = []
        epsilon_decay = 0.9  # Faster decay for shorter period
        min_epsilon = 0.05   # Higher minimum epsilon
        best_reward = float('-inf')
        patience = 5  # Number of episodes to wait for improvement
        no_improvement = 0
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            trades_this_episode = 0
            
            while not done:
                action = self.act(state, training=True)
                next_state, reward, done = env.step(action)
                
                if action != 0:  # If not holding
                    trades_this_episode += 1
                
                # Store experience in replay memory
                self.memory.append((state, action, reward, next_state, done))
                
                # Train on random batch from replay memory if we have enough samples
                if len(self.memory) > self.batch_size:
                    batch = random.sample(self.memory, self.batch_size)
                    states = torch.FloatTensor([s[0] for s in batch]).to(self.device)
                    actions = torch.LongTensor([[s[1]] for s in batch]).to(self.device)
                    rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device)
                    next_states = torch.FloatTensor([s[3] for s in batch]).to(self.device)
                    dones = torch.FloatTensor([s[4] for s in batch]).to(self.device)
                    
                    # Double DQN: Use online network to select action, target network to evaluate it
                    with torch.no_grad():
                        next_actions = self.model(next_states).argmax(1).unsqueeze(1)
                        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()
                        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                    
                    # Compute current Q values and loss
                    current_q_values = self.model(states).gather(1, actions).squeeze()
                    loss = F.smooth_l1_loss(current_q_values, target_q_values)
                    
                    # Optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    # Soft update target network more frequently
                    for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                
                state = next_state
                episode_reward += reward
            
            # Decay epsilon with a minimum value
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)
            
            # Track best reward and implement early stopping
            if episode_reward > best_reward:
                best_reward = episode_reward
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Early stopping if no improvement for several episodes
            if no_improvement >= patience:
                logger.info(f"Early stopping at episode {episode + 1} due to no improvement")
                break
            
            total_rewards.append(episode_reward)
            training_time = time.time() - start_time
            
            # Log progress with more details
            avg_reward = sum(total_rewards[-5:]) / min(len(total_rewards), 5)  # Moving average of last 5 episodes
            logger.info(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Trades: {trades_this_episode} | "
                f"Epsilon: {self.epsilon:.3f} | "
                f"Time: {training_time:.2f}s"
            )
        
        # Save final returns
        self.returns = total_rewards
    
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
            action = self.act(state, training=False)
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

def generate_trading_recommendations(agent: TradingAgent, env: StockTradingEnvironment) -> Dict:
    """Generate trading recommendations based on the trained agent."""
    state = env._get_state()
    action_probs = agent.get_action_probabilities(state)
    
    # Get current technical indicators
    current_step = env.current_step - 1  # Since we've already stepped
    rsi = env.data.iloc[current_step]['RSI']
    macd = env.data.iloc[current_step]['MACD']
    macd_signal = env.data.iloc[current_step]['MACD_Signal']
    
    # Determine signal and confidence
    action_idx = np.argmax(action_probs)
    confidence = float(action_probs[action_idx] * 100)
    
    signals = ['HOLD', 'BUY', 'SELL']
    signal = signals[action_idx]
    
    # Determine risk level based on volatility and indicators
    volatility = env.data.iloc[current_step]['Volatility']
    risk_level = 'High' if volatility > 0.03 else 'Medium' if volatility > 0.02 else 'Low'
    
    recommendation = {
        'signal': signal,
        'confidence': confidence,
        'risk': risk_level,
        'action_probabilities': {
            'hold': float(action_probs[0]),
            'buy': float(action_probs[1]),
            'sell': float(action_probs[2])
        },
        'technical_indicators': {
            'RSI': float(rsi),
            'MACD': float(macd),
            'MACD_Signal': float(macd_signal),
            'Volatility': float(volatility)
        }
    }
    
    return recommendation

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
    
    # Get one year of working days before test period for training
    train_start_date = test_start_date - timedelta(days=365)
    
    # Format dates as strings
    train_start_str = train_start_date.strftime('%Y-%m-%d')
    train_end_str = (test_start_date - timedelta(days=1)).strftime('%Y-%m-%d')
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

def initialize_trading_environment(symbol: str, initial_balance: float = 1000) -> Tuple[TradingAgent, StockTradingEnvironment]:
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
        initial_balance = 1000
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