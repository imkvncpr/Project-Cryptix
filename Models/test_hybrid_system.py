import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import time
import random

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(project_root))  # Go up one level

# Import HybridPredictor
from Models.Hybrid_Filter.Hybrid_Predictor import HybridPredictor

# Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def fetch_crypto_data(period='1y', use_cache=True, retry_delay=5, max_retries=3):
    """
    Fetch historical data for multiple cryptocurrencies with retry logic and caching
    
    Args:
        period (str): Time period to fetch ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        use_cache (bool): Whether to use cached data if available
        retry_delay (int): Initial delay in seconds before retrying after rate limit
        max_retries (int): Maximum number of retry attempts
    """
    print(f"Fetching {period} of historical data...")
    
    crypto_symbols = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'USDT': 'USDT-USD',
        'USDC': 'USDC-USD'
    }
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.getcwd(), "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    data = {}
    for crypto, symbol in crypto_symbols.items():
        # Define cache file path
        cache_file = os.path.join(cache_dir, f"{symbol}_{period}.csv")
        
        # Check for cached data if enabled
        if use_cache and os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                data[crypto] = df
                print(f"  ✓ Loaded cached data for {crypto} ({len(df)} rows)")
                continue  # Skip to next crypto
            except Exception as e:
                print(f"  ✗ Error loading cached data for {crypto}: {e}")
        
        # If no cache or loading cache failed, fetch from Yahoo Finance
        print(f"Fetching {crypto} data...")
        
        # Implement retry logic with exponential backoff
        retries = 0
        current_delay = retry_delay
        success = False
        
        while retries <= max_retries and not success:
            try:
                # Add a small random delay to help avoid rate limits
                jitter = random.uniform(0.5, 2.0)
                time.sleep(jitter)
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                
                # Ensure required columns exist
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_columns) and not df.empty:
                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    
                    # Save to cache
                    if use_cache:
                        df.to_csv(cache_file)
                        print(f"  ✓ Saved {crypto} data to cache")
                    
                    data[crypto] = df
                    print(f"  ✓ {len(df)} rows fetched for {crypto}")
                    success = True
                else:
                    missing = [col for col in required_columns if col not in df.columns]
                    if df.empty:
                        missing = ["Empty dataframe returned"]
                    print(f"  ✗ Missing data for {crypto}: {missing}")
                    retries += 1
                    
                # Add a significant delay before fetching the next cryptocurrency
                if not crypto == list(crypto_symbols.keys())[-1]:  # If not the last crypto
                    delay = random.uniform(10, 15)  # Random delay between 10-15 seconds
                    print(f"  → Waiting {delay:.1f} seconds before next request...")
                    time.sleep(delay)
            
            except Exception as e:
                err_msg = str(e)
                retries += 1
                
                if "Too Many Requests" in err_msg and retries <= max_retries:
                    print(f"  ⚠ Rate limited for {crypto}. Retry {retries}/{max_retries} in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= 2  # Exponential backoff
                else:
                    print(f"  ✗ Error fetching {crypto}: {e}")
                    if retries <= max_retries:
                        print(f"  → Retry {retries}/{max_retries} in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= 1.5
                    else:
                        break
    
    return data

def download_data_bulk(period='1y'):
    """
    Alternative approach to download all data at once using yf.download
    with multiple tickers, which may avoid some rate limiting issues.
    """
    print(f"Bulk downloading {period} of historical data...")
    
    tickers = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'USDC-USD']
    ticker_str = ' '.join(tickers)
    
    try:
        # Download all at once
        df = yf.download(ticker_str, period=period, group_by='ticker')
        
        # Process the multi-level dataframe
        data = {}
        for ticker in tickers:
            if ticker in df.columns:
                # Extract data for this ticker
                ticker_data = df[ticker].copy()
                if not ticker_data.empty:
                    # Rename columns to lowercase
                    ticker_data.columns = [col.lower() for col in ticker_data.columns]
                    # Map to crypto name
                    crypto = ticker.split('-')[0]
                    data[crypto] = ticker_data
                    print(f"  ✓ {len(ticker_data)} rows fetched for {crypto}")
            else:
                print(f"  ✗ No data found for {ticker}")
                
        return data
    except Exception as e:
        print(f"  ✗ Error in bulk download: {e}")
        return {}

def visualize_predictions(data, results, crypto='BTC'):
    """
    Create visualizations of predictions and signals
    """
    if crypto not in results:
        print(f"No results for {crypto}")
        return
    
    # Get data for the selected crypto
    df = data[crypto]
    result = results[crypto]
    
    # Prepare data for plotting
    dates = df.index[-len(result['prediction']):]
    actual = df['close'].values[-len(result['prediction']):]
    predicted = result['prediction']
    signals = result['final_signal']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Price and Predictions
    ax1.plot(dates, actual, label='Actual Price', color='blue')
    ax1.plot(dates, predicted, label='Predicted Price', color='green', linestyle='--')
    
    # Highlight buy signals
    buy_dates = [date for date, signal in zip(dates, signals) if signal == 1]
    buy_prices = [price for price, signal in zip(actual, signals) if signal == 1]
    ax1.scatter(buy_dates, buy_prices, color='red', marker='^', s=100, label='Buy Signal')
    
    # Format date axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    
    # Add labels and legend
    ax1.set_title(f'{crypto} Price Prediction and Signals')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Technical Indicators
    try:
        tech_signals = result['technical_signals']
        
        # RSI Signal
        if 'rsi_signal' in tech_signals:
            rsi_signal = tech_signals['rsi_signal']
            ax2.plot(dates, rsi_signal, label='RSI Signal', color='purple')
        
        # MACD Signal
        if 'macd_signal' in tech_signals:
            macd_signal = tech_signals['macd_signal']
            ax2.plot(dates, macd_signal, label='MACD Signal', color='orange')
        
        # Volume Signal
        if 'volume_signal' in tech_signals:
            volume_signal = tech_signals['volume_signal']
            ax2.plot(dates, volume_signal, label='Volume Signal', color='brown')
        
        # Combined Signal
        ax2.plot(dates, signals, label='Final Signal', color='red', linewidth=2)
        
        ax2.set_ylabel('Signal')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True)
    except:
        ax2.text(0.5, 0.5, 'Technical Signal Data Not Available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{crypto}_prediction_results.png")
    print(f"Visualization saved as {crypto}_prediction_results.png")
    plt.close()

def calculate_performance(data, results, crypto='BTC'):
    """
    Calculate performance metrics for predictions and signals
    """
    if crypto not in results:
        return None
    
    df = data[crypto]
    result = results[crypto]
    
    # Get data
    actual = df['close'].values[-len(result['prediction']):]
    predicted = result['prediction']
    signals = result['final_signal']
    
    # Calculate prediction metrics
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Calculate signal performance
    # Assume buy and hold for 1 day
    returns = []
    for i in range(len(signals) - 1):
        if signals[i] == 1:  # Buy signal
            daily_return = (actual[i+1] - actual[i]) / actual[i] * 100
            returns.append(daily_return)
    
    # Calculate signal metrics
    if returns:
        avg_return = np.mean(returns)
        win_rate = np.sum(np.array(returns) > 0) / len(returns) * 100
        max_return = np.max(returns) if returns else 0
        min_return = np.min(returns) if returns else 0
    else:
        avg_return = win_rate = max_return = min_return = 0
    
    return {
        'crypto': crypto,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'signal_count': np.sum(signals),
        'avg_return': avg_return,
        'win_rate': win_rate,
        'max_return': max_return,
        'min_return': min_return
    }

def main():
    print("\n===== Testing Hybrid Prediction System =====\n")
    
    # 1. Try to fetch historical data with retry logic and caching
    data = fetch_crypto_data(period='1y', use_cache=True, retry_delay=5, max_retries=3)
    
    # 2. If the standard approach fails, try bulk download
    if not data:
        print("\nIndividual fetching failed. Trying bulk download...\n")
        data = download_data_bulk(period='1y')
    
    if not data:
        print("No data fetched. Exiting.")
        return
    
    print(f"\nData fetched successfully for {len(data)} cryptocurrencies.\n")
    
    # 3. Initialize hybrid predictor
    print("Initializing Hybrid Predictor...")
    predictor = HybridPredictor(look_back=60)
    
    # 4. Generate predictions and signals
    print("Generating predictions and signals...")
    results = predictor.predict(data)
    
    # 5. Generate report
    print("\n===== Prediction Report =====\n")
    report = predictor.generate_report(results)
    print(report)
    
    # 6. Visualize results
    print("\n===== Creating Visualizations =====\n")
    for crypto in results:
        visualize_predictions(data, results, crypto)
    
    # 7. Calculate performance
    print("\n===== Performance Metrics =====\n")
    performance_metrics = []
    for crypto in results:
        metrics = calculate_performance(data, results, crypto)
        if metrics:
            performance_metrics.append(metrics)
    
    # Create performance DataFrame
    if performance_metrics:
        performance_df = pd.DataFrame(performance_metrics)
        print(performance_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        # Save performance metrics
        performance_df.to_csv("hybrid_system_performance.csv", index=False)
        print("\nPerformance metrics saved to hybrid_system_performance.csv")
    
    print("\n===== Test Complete =====\n")

if __name__ == "__main__":
    main()