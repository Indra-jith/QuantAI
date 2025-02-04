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

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/process_stock', methods=['POST'])
def process_stock():
    """Process the stock analysis request."""
    try:
        symbol = request.form.get('symbol')
        if not symbol:
            flash('Please enter a valid stock symbol')
            return redirect(url_for('index'))
            
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
