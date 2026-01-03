from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import json
from Main_code import initialize_trading_environment, generate_trading_recommendations, save_trading_results, StockDataProcessor
import logging
import re
import requests

app = Flask(__name__)
app.secret_key = 'dev'  # Simple secret key for development

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API key directly in code (since it's public)
API_KEY = 'STHK350R8VAMIH2Q'

# Famous stocks list directly in code
STOCK_LIST = [
    {"symbol": "AAPL", "name": "Apple Inc"},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc (Google)"},
    {"symbol": "AMZN", "name": "Amazon.com Inc"},
    {"symbol": "META", "name": "Meta Platforms Inc (Facebook)"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "TSLA", "name": "Tesla Inc"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co"},
    {"symbol": "V", "name": "Visa Inc"},
    {"symbol": "WMT", "name": "Walmart Inc"},
    {"symbol": "MA", "name": "Mastercard Inc"},
    {"symbol": "PG", "name": "Procter & Gamble Company"},
    {"symbol": "DIS", "name": "Walt Disney Co"},
    {"symbol": "NFLX", "name": "Netflix Inc"},
    {"symbol": "CSCO", "name": "Cisco Systems Inc"},
    {"symbol": "KO", "name": "Coca-Cola Company"},
    {"symbol": "PEP", "name": "PepsiCo Inc"},
    {"symbol": "ADBE", "name": "Adobe Inc"},
    {"symbol": "INTC", "name": "Intel Corporation"},
    {"symbol": "AMD", "name": "Advanced Micro Devices Inc"},
    {"symbol": "PYPL", "name": "PayPal Holdings Inc"},
    {"symbol": "CMCSA", "name": "Comcast Corporation"},
    {"symbol": "NKE", "name": "Nike Inc"},
    {"symbol": "VZ", "name": "Verizon Communications Inc"},
    {"symbol": "T", "name": "AT&T Inc"},
    {"symbol": "XOM", "name": "Exxon Mobil Corporation"},
    {"symbol": "BAC", "name": "Bank of America Corp"},
    {"symbol": "HD", "name": "Home Depot Inc"},
    {"symbol": "MCD", "name": "McDonald's Corporation"},
    {"symbol": "CRM", "name": "Salesforce Inc"},
    {"symbol": "UBER", "name": "Uber Technologies Inc"},
    {"symbol": "BA", "name": "Boeing Company"}
]

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/get_stock_data')
def get_stock_data():
    """Return the stock list."""
    return jsonify(STOCK_LIST)

@app.route('/results')
def results():
    """Display the results page."""
    try:
        results_path = os.path.join(os.path.dirname(__file__), 'trading_results.json')
        
        if not os.path.exists(results_path):
            logger.error("Results file not found")
            flash('No results available')
            return redirect(url_for('index'))
        
        try:
            with open(results_path, 'r') as f:
                file_content = f.read()
                if not file_content.strip():
                    logger.error("Results file is empty")
                    flash('No results available')
                    return redirect(url_for('index'))
                results = json.loads(file_content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in results file: {str(e)}")
            flash('Error loading results')
            return redirect(url_for('index'))
        
        # Validate required fields
        required_fields = ['symbol', 'final_portfolio_value', 'initial_balance', 'model_performance']
        missing_fields = [field for field in required_fields if field not in results]
        if missing_fields:
            logger.error(f"Missing required fields in results: {missing_fields}")
            flash('Invalid results data')
            return redirect(url_for('index'))
        
        # Ensure all required fields exist with defaults
        if 'model_performance' not in results:
            results['model_performance'] = {}
        
        model_perf = results['model_performance']
        model_perf.setdefault('total_trades', 0)
        model_perf.setdefault('total_profit', 0)
        
        # Ensure required fields exist
        results.setdefault('final_portfolio_value', results.get('testing', {}).get('final_value', 10000))
        results.setdefault('total_return_pct', results.get('testing', {}).get('return', 0) * 100)
        results.setdefault('initial_balance', 10000)
        results.setdefault('trading_history', [])
        
        logger.info("Successfully loaded results for display")
        return render_template('results.html', results=results)
        
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        flash('Error loading results')
        return redirect(url_for('index'))

@app.route('/process_stock', methods=['POST'])
def process_stock():
    """Process the stock analysis request."""
    try:
        symbol = request.form.get('symbol', '').strip().upper()
        if not symbol:
            flash('Please enter a valid stock symbol')
            return redirect(url_for('index'))
        
        # Validate symbol format
        if not re.match(r'^[A-Z]{1,5}$', symbol):
            flash('Invalid stock symbol format')
            return redirect(url_for('index'))
        
        os.environ['ALPHA_VANTAGE_API_KEY'] = API_KEY
        
        try:
            # Initialize environments
            agent, train_env, test_env = initialize_trading_environment(symbol)
            
            # Train the agent
            train_results = agent.evaluate_model(train_env)
            logger.info(f"Training completed for {symbol}")
            
            # Test the agent
            test_results = agent.evaluate_model(test_env)
            logger.info(f"Testing completed for {symbol}")
            
            # Generate recommendations
            recommendation = generate_trading_recommendations(agent, test_env)
            
            # Save results
            save_trading_results(symbol, train_results, test_results, recommendation)
            logger.info(f"Results saved for {symbol}")
            
            return redirect(url_for('results'))
            
        except ValueError as e:
            logger.error(f"Value error processing {symbol}: {str(e)}")
            flash(str(e))
            return redirect(url_for('index'))
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API error for {symbol}: {str(e)}")
            flash('Error accessing stock data. Please try again later.')
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"Unexpected error processing {symbol}: {str(e)}")
            flash('An unexpected error occurred. Please try again later.')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        flash('A system error occurred. Please try again later.')
        return redirect(url_for('index'))

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors by redirecting to index."""
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
