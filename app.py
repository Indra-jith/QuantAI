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
        # Get symbol from form
        symbol = request.form.get('symbol', '').strip().upper()
        if not symbol:
            logger.warning("Empty stock symbol provided")
            return jsonify({'error': 'Please select a stock from the suggestions'}), 400

        # Validate symbol exists in our list
        stock_list = load_stock_data()
        valid_symbols = {stock['symbol'] for stock in stock_list}
        if symbol not in valid_symbols:
            logger.warning(f"Invalid stock symbol provided: {symbol}")
            return jsonify({'error': 'Please select a valid stock symbol from the suggestions'}), 400
        
        try:
            logger.info(f"Processing stock data for {symbol}")
            
            # Initialize processor first
            processor = StockDataProcessor(api_key=API_KEY)
            
            # Get date ranges
            date_ranges = get_date_ranges()
            logger.info(f"Date ranges - Training: {date_ranges['train']}, Testing: {date_ranges['test']}")
            
            # Get training data first
            logger.info(f"Fetching training data for {symbol}")
            train_data = processor.get_data_for_period(
                symbol, 
                date_ranges['train'][0], 
                date_ranges['train'][1]
            )
            if len(train_data) < 20:
                return jsonify({'error': f'Insufficient training data for {symbol}. Need at least 20 days.'}), 400
            
            # Get testing data
            logger.info(f"Fetching testing data for {symbol}")
            test_data = processor.get_data_for_period(
                symbol, 
                date_ranges['test'][0], 
                date_ranges['test'][1]
            )
            if len(test_data) < 5:
                return jsonify({'error': f'Insufficient testing data for {symbol}. Need at least 5 days.'}), 400
            
            # Initialize environments
            logger.info("Initializing training environment")
            train_env = StockTradingEnvironment(train_data, initial_balance=1000)
            
            logger.info("Initializing testing environment")
            test_env = StockTradingEnvironment(test_data, initial_balance=1000)
            
            # Initialize agent
            logger.info("Initializing trading agent")
            agent = TradingAgent(state_size=train_env.state_size, action_size=train_env.action_size)
            
            # Train the model
            logger.info(f"Training model for {symbol}")
            agent.train(train_env, episodes=20)
            
            # Evaluate on test data
            logger.info(f"Evaluating model performance")
            training_results = agent.evaluate_model(test_env)
            
            # Generate recommendations
            logger.info(f"Generating trading recommendations")
            recommendation = generate_trading_recommendations(agent, test_env)
            
            # Convert numpy values to Python native types
            def convert_to_native(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: convert_to_native(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_native(item) for item in obj]
                return obj
            
            # Calculate metrics
            final_value = float(training_results.get('final_value', 0))
            initial_value = 1000.00
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            # Prepare results with converted values
            results = {
                'symbol': symbol,
                'training_return': float(total_return),
                'final_portfolio_value': float(final_value),
                'recommendation': convert_to_native(recommendation),
                'trading_history': convert_to_native(training_results.get('trading_history', [])),
                'model_performance': {
                    'avg_reward': float(training_results.get('avg_reward', 0)),
                    'final_epsilon': float(training_results.get('final_epsilon', 0)),
                    'training_episodes': 20,
                    'sharpe_ratio': float(training_results.get('sharpe_ratio', 0)),
                    'max_drawdown': float(training_results.get('max_drawdown', 0))
                },
                'dates': {
                    'training_start': date_ranges['train'][0],
                    'training_end': date_ranges['train'][1],
                    'testing_start': date_ranges['test'][0],
                    'testing_end': date_ranges['test'][1]
                }
            }
            
            # Save results to file
            logger.info(f"Saving results for {symbol}")
            results_file = os.path.join(os.path.dirname(__file__), 'trading_results.json')
            with open(results_file, 'w') as f:
                json.dump(convert_to_native(results), f, indent=4)
            
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
