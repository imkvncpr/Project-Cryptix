import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(project_root))  # Go up one level

# Import HybridPredictor
from Models.Hybrid_Filter.Hybrid_Predictor import HybridPredictor

# Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def fetch_crypto_data(period='1y'):
    """
    Fetch historical data for multiple cryptocurrencies
    """
    print(f"Fetching {period} of historical data...")
    
    crypto_symbols = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'USDT': 'USDT-USD',
        'USDC': 'USDC-USD'
    }
    
    data = {}
    for crypto, symbol in crypto_symbols.items():
        try:
            print(f"Fetching {crypto} data...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df.columns for col in required_columns):
                # Rename columns to match expected format
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                data[crypto] = df
                print(f"  ✓ {len(df)} rows fetched for {crypto}")
            else:
                missing = [col for col in required_columns if col not in df.columns]
                print(f"  ✗ Missing columns for {crypto}: {missing}")
        except Exception as e:
            print(f"  ✗ Error fetching {crypto}: {e}")
    
    return data

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
    
    # 1. Fetch historical data
    data = fetch_crypto_data(period='1y')
    
    if not data:
        print("No data fetched. Exiting.")
        return
    
    print(f"\nData fetched successfully for {len(data)} cryptocurrencies.\n")
    
    # 2. Initialize hybrid predictor
    print("Initializing Hybrid Predictor...")
    predictor = HybridPredictor(look_back=60)
    
    # 3. Generate predictions and signals
    print("Generating predictions and signals...")
    results = predictor.predict(data)
    
    # 4. Generate report
    print("\n===== Prediction Report =====\n")
    report = predictor.generate_report(results)
    print(report)
    
    # 5. Visualize results
    print("\n===== Creating Visualizations =====\n")
    for crypto in results:
        visualize_predictions(data, results, crypto)
    
    # 6. Calculate performance
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