# Stock Trading Bot Using Reinforcement Learning

This project implements a **stock trading bot** that utilizes **Reinforcement Learning (RL)** to make trading decisions based on historical stock market data. The bot is trained using **Deep Q-Learning (DQN)** and executes trades based on market indicators.

## 📂 Project Overview
- **Objective**: Build a deep reinforcement learning-based trading agent to optimize stock trading strategies.
- **Technology Stack**:
  - **Python** (for data processing and modeling)
  - **Flask** (for web-based interaction with the trading bot)
  - **PyTorch** (for implementing DQN-based reinforcement learning)
  - **Alpha Vantage API** (for fetching real-time stock market data)
- **Main Components**:
  - `Main_code.py`: Core implementation of stock data processing, trading environment, and reinforcement learning model.
  - `app.py`: Flask-based web interface to interact with the trading bot.

## 🚀 Features
- **Data Collection**: Fetches real-time stock data from Alpha Vantage API.
- **Technical Indicators**: Uses RSI, MACD, Bollinger Bands, Moving Averages, and more.
- **Reinforcement Learning**: Implements **Deep Q-Learning (DQN)** for optimal trading decisions.
- **Portfolio Management**: Tracks account balance, holdings, and trading history.
- **Web Interface**: Allows users to select stocks and visualize trading results.

## 📦 Installation
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/stock-trading-bot.git
   cd stock-trading-bot
   ```
2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the Flask App**:
   ```sh
   python app.py
   ```

## 📊 How It Works
1. **Load Stock Data**: Fetches stock prices and indicators using `Alpha Vantage API`.
2. **Train the Trading Agent**: Uses past stock data to train the RL model.
3. **Make Predictions**: The model generates trading signals (`BUY`, `SELL`, `HOLD`).
4. **Simulated Trading**: Executes trades in a simulated environment and evaluates performance.
5. **Results Analysis**: Displays performance metrics, including Sharpe Ratio, Portfolio Value, and Total Returns.

## 🛠 API Endpoints
- `/get_stock_data` → Fetches available stock symbols.
- `/process_stock` → Processes stock data, trains the model, and generates trading signals.
- `/results` → Displays trading results and recommendations.

## 🤝 Contributing
- Fork the repository and submit pull requests with improvements.
- Open issues for bug reports and feature requests.

---
### 🔗 Useful Links
- [Alpha Vantage API](https://www.alphavantage.co/)
- [Reinforcement Learning in Trading](https://www.kaggle.com/search?q=reinforcement+learning+trading)
