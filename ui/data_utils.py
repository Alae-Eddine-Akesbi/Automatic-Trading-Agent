import pandas as pd

def discretize_return(ret):
    if ret > 0.01:
        return 2
    elif ret < -0.01:
        return 0
    else:
        return 1

def feature_engineering(data):
    data['Return'] = data['Adj Close'].pct_change()
    data['MA5'] = data['Adj Close'].rolling(5).mean()
    data['MA20'] = data['Adj Close'].rolling(20).mean()
    data['MA_Signal'] = (data['MA5'] > data['MA20']).astype(int)
    delta = data['Adj Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    rolling_mean = data['Adj Close'].rolling(20).mean()
    rolling_std = data['Adj Close'].rolling(20).std()
    data['Bollinger_Upper'] = rolling_mean + 2 * rolling_std
    data['Bollinger_Lower'] = rolling_mean - 2 * rolling_std
    exp1 = data['Adj Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Adj Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Volume_Norm'] = (data['Volume'] - data['Volume'].rolling(20).mean()) / data['Volume'].rolling(20).std()
    data['Return_Class'] = data['Return'].apply(discretize_return)
    data = data.dropna()
    # Ensure index is datetime for all downstream use
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
        else:
            data.index = pd.to_datetime(data.index)
    return data
