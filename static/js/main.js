document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('#stockForm');
    const loadingDiv = document.querySelector('#loading');
    const loadingOverlay = document.querySelector('.loading-overlay');
    
    function showLoading() {
        loadingDiv.classList.remove('hidden');
        // Add typing animation to each message
        const messages = loadingDiv.querySelectorAll('.typing');
        messages.forEach((msg, index) => {
            msg.style.animationDelay = `${index * 2}s`;
            msg.classList.add('typing-animation');
        });
    }
    
    function hideLoading() {
        loadingDiv.classList.add('hidden');
        // Reset animations
        const messages = loadingDiv.querySelectorAll('.typing');
        messages.forEach(msg => {
            msg.classList.remove('typing-animation');
        });
    }
    
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const symbol = document.querySelector('input[name="symbol"]').value.trim();
            if (!symbol) {
                alert('Please enter a stock symbol');
                return;
            }
            
            // Show terminal loading
            showLoading();
            
            try {
                const formData = new FormData(this);
                const response = await fetch('/process_stock', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'Accept': '*/*'  // Accept any response type
                    }
                });
                
                // Handle redirects
                if (response.redirected) {
                    window.location.href = response.url;
                    return;
                }
                
                // Try to parse JSON if available
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const data = await response.json();
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    if (data.redirect) {
                        window.location.href = data.redirect;
                        return;
                    }
                }
                
                // If we get here, follow the response URL
                window.location.href = response.url;
                
            } catch (error) {
                console.error('Error:', error);
                alert(error.message || 'An error occurred while processing your request');
                hideLoading();
            }
        });
    }
});

// Terminal output simulation
const terminalOutput = document.getElementById('terminal-output');
const loadingContainer = document.getElementById('loading');

function addTerminalLine(text) {
    const line = document.createElement('div');
    line.className = 'terminal-line';
    line.innerHTML = `<span class="prompt">$</span> ${text}`;
    terminalOutput.appendChild(line);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

// Load stock symbols and company names
let stockData = [];

// Debug flag
const DEBUG = true;

function log(...args) {
    if (DEBUG) {
        console.log(...args);
    }
}

// Load stock data when page loads
document.addEventListener('DOMContentLoaded', function() {
    log('Loading stock data...');
    fetch('/get_stock_data')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load stock data');
            }
            return response.json();
        })
        .then(data => {
            stockData = data;
            log(`Loaded ${stockData.length} stock symbols`);
        })
        .catch(error => {
            console.error('Error loading stock data:', error);
            alert('Failed to load stock data. Please refresh the page.');
        });
});

// Handle company name input and suggestions
const companyNameInput = document.getElementById('companyName');
const symbolInput = document.getElementById('symbol');
const suggestionsBox = document.getElementById('suggestions');

companyNameInput.addEventListener('input', function() {
    const query = this.value.toLowerCase();
    
    if (query.length < 2) {
        suggestionsBox.style.display = 'none';
        return;
    }

    const matches = stockData.filter(item => 
        item.name.toLowerCase().includes(query) || 
        item.symbol.toLowerCase().includes(query)
    ).slice(0, 10);  // Limit to 10 suggestions

    if (matches.length > 0) {
        suggestionsBox.innerHTML = matches.map(item => `
            <div class="suggestion-item" data-symbol="${item.symbol}">
                <strong>${item.symbol}</strong> - ${item.name}
            </div>
        `).join('');
        suggestionsBox.style.display = 'block';
    } else {
        suggestionsBox.style.display = 'none';
    }
});

// Handle suggestion selection
suggestionsBox.addEventListener('click', function(e) {
    const item = e.target.closest('.suggestion-item');
    if (item) {
        const symbol = item.dataset.symbol;
        const name = item.textContent.split(' - ')[1];
        
        companyNameInput.value = name;
        symbolInput.value = symbol;
        suggestionsBox.style.display = 'none';
        
        // Get the form and submit it
        const form = document.getElementById('stockForm');
        const formData = new FormData(form);
        
        // Show loading animation
        const loadingElement = document.getElementById('loading');
        loadingElement.classList.remove('hidden');
        
        // Make the API call
        fetch(form.action, {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': '*/*'  // Accept any response type
            }
        })
        .then(response => {
            // Check if we got redirected
            if (response.redirected) {
                window.location.href = response.url;
                return;
            }
            
            // Try to parse as JSON if possible
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return response.json().then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    if (data.redirect) {
                        window.location.href = data.redirect;
                    }
                });
            }
            
            // If we get here, just follow the response
            window.location.href = response.url;
        })
        .catch(error => {
            console.error('Error:', error);
            loadingElement.classList.add('hidden');
            alert('An error occurred. Please try again.');
        });
    }
});

// Handle form submission and loading animation
function analyzeStock(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    
    // Check if we have a symbol
    const symbol = formData.get('symbol');
    if (!symbol) {
        alert('Please select a stock from the suggestions');
        return;
    }
    
    // Show loading animation
    const loadingElement = document.getElementById('loading');
    loadingElement.classList.remove('hidden');
    
    // Make the API call
    fetch(form.action, {
        method: 'POST',
        body: formData,
        headers: {
            'Accept': '*/*'  // Accept any response type
        }
    })
    .then(response => {
        // Check if we got redirected
        if (response.redirected) {
            window.location.href = response.url;
            return;
        }
        
        // Try to parse as JSON if possible
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return response.json().then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                if (data.redirect) {
                    window.location.href = data.redirect;
                }
            });
        }
        
        // If we get here, just follow the response
        window.location.href = response.url;
    })
    .catch(error => {
        console.error('Error:', error);
        loadingElement.classList.add('hidden');
        alert('An error occurred. Please try again.');
    });
}

// Hide suggestions when clicking outside
document.addEventListener('click', function(e) {
    if (!e.target.closest('.input-group')) {
        suggestionsBox.style.display = 'none';
    }
});

// Add input validation
document.getElementById('symbol').addEventListener('input', function(e) {
    this.value = this.value.replace(/[^A-Za-z]/g, '').toUpperCase();
});

function resetForm() {
    // Clear the input
    document.getElementById('symbol').value = '';
    
    // Show the search container
    document.querySelector('.search-container').classList.remove('hidden');
    
    // Hide loading and results
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('results').classList.add('hidden');
}
