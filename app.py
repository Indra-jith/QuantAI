from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, current_app
from Main_code import StockDataProcessor, StockTradingEnvironment, TradingAgent, get_date_ranges, generate_trading_recommendations, initialize_trading_environment
import logging
import traceback
import json
from datetime import datetime
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'secret_key_here'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API key
API_KEY = 'BF24E50ARR9HZEP0'  # Alpha Vantage API key
app.config['ALPHA_VANTAGE_API_KEY'] = API_KEY

# Load stock data from CSV
def load_stock_data():
    try:
        df = pd.read_csv('LIST.csv')
        data = df[['symbol', 'name']].dropna().to_dict('records')
        logger.info(f"Loaded {len(data)} stock symbols")
        return data
    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
        return []

@app.route('/')
def index():
    """Render the landing page."""
    return render_template('index.html')

@app.route('/get_stock_data')
def get_stock_data():
    """Load and return stock data from CSV."""
    try:
        data = load_stock_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error loading stock data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_stock', methods=['POST'])
def process_stock():
    """Process stock data and generate trading signals."""
    try:
        symbol = request.form.get('symbol', '').strip().upper()
        if not symbol:
            return jsonify({'error': 'Please select a stock from the suggestions'}), 400

        stock_list = load_stock_data()
        valid_symbols = {stock['symbol'] for stock in stock_list}
        if symbol not in valid_symbols:
            return jsonify({'error': 'Please select a valid stock symbol from the suggestions'}), 400
        
        try:
            logger.info(f"Processing stock data for {symbol}")
            processor = StockDataProcessor(api_key=API_KEY)
            
            # Get date ranges
            date_ranges = get_date_ranges()
            
            # Get training data
            logger.info(f"Fetching training data for {symbol}")
            train_data = processor.get_data_for_period(
                symbol, 
                date_ranges['train'][0], 
                date_ranges['train'][1]
            )
            
            if len(train_data) < 200:
                return jsonify({'error': f'Insufficient historical data for {symbol}. Need at least 200 trading days.'}), 400
            
            # Get testing data
            logger.info(f"Fetching testing data for {symbol}")
            test_data = processor.get_data_for_period(
                symbol, 
                date_ranges['test'][0], 
                date_ranges['test'][1]
            )
            
            if len(test_data) < 5:
                return jsonify({'error': f'Insufficient recent data for {symbol}. Need at least 5 trading days.'}), 400
            
            # Set initial balance to 100
            initial_balance = 100.0
            
            logger.info("Initializing environments")
            train_env = StockTradingEnvironment(train_data, initial_balance=initial_balance)
            test_env = StockTradingEnvironment(test_data, initial_balance=initial_balance)
            
            # Initialize and train agent
            logger.info("Training model")
            agent = TradingAgent(
                state_size=train_env.state_size,
                action_size=train_env.action_size,
                learning_rate=0.001,
                gamma=0.95,
                epsilon=1.0,
                epsilon_min=0.1,
                epsilon_decay=0.995
            )
            
            train_env.reset()
            agent.train(train_env, episodes=20)
            
            # Test the model and get recommendations
            logger.info("Testing model and generating recommendations")
            test_env.reset()
            recommendation = generate_trading_recommendations(agent, test_env)
            
            # Run test episode to generate trading history
            state = test_env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done = test_env.step(action)
                state = next_state
            
            # Calculate performance metrics
            portfolio_values = np.array([trade['portfolio_value'] for trade in test_env.trading_history] if test_env.trading_history else [initial_balance])
            
            if len(portfolio_values) > 1:
                daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
                volatility = float(np.std(daily_returns) * np.sqrt(252))
                sharpe_ratio = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252))
                
                # Calculate max drawdown properly
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (peak - portfolio_values) / peak
                max_drawdown = float(np.max(drawdown))
            else:
                daily_returns = np.array([0])
                volatility = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
            
            # Prepare results
            results = {
                'symbol': symbol,
                'initial_balance': initial_balance,
                'final_portfolio_value': float(test_env._portfolio_value),
                'total_return_pct': ((test_env._portfolio_value - initial_balance) / initial_balance) * 100,
                'recommendation': {
                    'action': recommendation['action'],
                    'confidence': recommendation['confidence'],
                    'reason': recommendation['reason'],
                    'current_price': float(test_data['close'].iloc[-1]),
                    'probabilities': {
                        'buy': float(recommendation['probabilities']['buy']),
                        'sell': float(recommendation['probabilities']['sell']),
                        'hold': float(recommendation['probabilities']['hold'])
                    },
                    'technical_indicators': {
                        'RSI': float(recommendation['technical_indicators']['RSI']),
                        'MACD': float(recommendation['technical_indicators']['MACD']),
                        'MACD_Signal': float(recommendation['technical_indicators']['MACD_Signal']),
                        'EMA5': float(recommendation['technical_indicators']['EMA5']),
                        'EMA20': float(recommendation['technical_indicators']['EMA20'])
                    }
                },
                'trading_history': [
                    {
                        'date': trade['date'],
                        'action': trade['action'],
                        'shares': float(trade['shares']),
                        'price': float(trade['price']),
                        'value': float(trade.get('cost', trade.get('revenue', 0))),
                        'profit': float(trade.get('profit', 0)) if 'profit' in trade else None,
                        'portfolio_value': float(trade['portfolio_value'])
                    }
                    for trade in test_env.trading_history
                ],
                'model_performance': {
                    'total_trades': test_env.total_trades,
                    'profitable_trades': test_env.profitable_trades,
                    'win_rate': float(test_env.profitable_trades / max(test_env.total_trades, 1) * 100),
                    'total_profit': float(test_env.total_profit),
                    'sharpe_ratio': sharpe_ratio,
                    'volatility': volatility,
                    'max_drawdown': max_drawdown,
                    'training_episodes': 20
                },
                'market_analysis': {
                    'trend': 'bullish' if test_data['EMA5'].iloc[-1] > test_data['EMA20'].iloc[-1] else 'bearish',
                    'volatility': volatility,
                    'rsi': float(test_data['RSI'].iloc[-1]),
                    'volume_trend': 'increasing' if test_data['volume'].iloc[-1] > test_data['Volume_MA5'].iloc[-1] else 'decreasing'
                },
                'dates': date_ranges
            }
            
            # Save results
            results_file = os.path.join(os.path.dirname(__file__), 'trading_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"Successfully processed {symbol}")
            return jsonify({'redirect': url_for('results')})
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Error processing stock data: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in process_stock: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Error processing stock data. Please try again.'}), 500

@app.route('/results')
def results():
    """Display trading results."""
    try:
        # Read results from file
        results_file = os.path.join(os.path.dirname(__file__), 'trading_results.json')
        if not os.path.exists(results_file):
            return render_template('error.html', error='No trading results available. Please analyze a stock first.')
            
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        # Format dates
        for period in ['training_start', 'training_end', 'testing_start', 'testing_end']:
            if period in results.get('dates', {}):
                date_str = results['dates'][period]
                results['dates'][period] = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        
        return render_template('results.html', results=results)
        
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('error.html', error='Error displaying results. Please try again.')

if __name__ == '__main__':
    app.run(debug=True)
