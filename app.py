from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
from Main_code import initialize_trading_environment, generate_trading_recommendations, save_trading_results
import logging

app = Flask(__name__)
app.secret_key = 'dev'  # Simple secret key for development

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API key directly in code (since it's public)
API_KEY = 'BF24E50ARR9HZEP0'

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

@app.route('/process_stock', methods=['POST'])
def process_stock():
    """Process the stock analysis request."""
    try:
        symbol = request.form.get('symbol')
        if not symbol:
            flash('Please enter a valid stock symbol')
            return redirect(url_for('index'))
            
        # Make API key available to Main_code.py
        os.environ['ALPHA_VANTAGE_API_KEY'] = API_KEY
        
        # Initialize environment and agent
        agent, env = initialize_trading_environment(symbol)
        
        # Train the agent
        results = agent.evaluate_model(env)
        
        # Generate recommendations
        recommendation = generate_trading_recommendations(agent, env)
        
        # Save results
        save_trading_results(symbol, results, recommendation)
        
        return jsonify(recommendation)
        
    except Exception as e:
        logger.error(f"Error processing stock: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors by redirecting to index."""
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
