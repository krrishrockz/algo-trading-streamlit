import pandas as pd
import yfinance as yf
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import gymnasium as gym

    
import feedparser
from datetime import datetime

import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, filename="ml_strategy.log")

def plot_trade_signals(df, ticker, model_name, chart_type="plotly"):
    """
    Plots trade signals with historical prices.

    Args:
        df (pd.DataFrame): DataFrame with 'Date', 'Close', 'Signal', 'Prediction'.
        ticker (str): Stock ticker.
        model_name (str): Name of the ML model.
        chart_type (str): 'plotly' or 'matplotlib'.

    Returns:
        fig: Plotly or Matplotlib figure.
    """
    if chart_type == "plotly":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close Price", line=dict(color="blue")))
        buy_signals = df[df["Prediction"] == 1]
        sell_signals = df[df["Prediction"] == -1]
        fig.add_trace(go.Scatter(x=buy_signals["Date"], y=buy_signals["Close"], name="Buy Signal",
                                 mode="markers", marker=dict(symbol="triangle-up", size=10, color="green")))
        fig.add_trace(go.Scatter(x=sell_signals["Date"], y=sell_signals["Close"], name="Sell Signal",
                                 mode="markers", marker=dict(symbol="triangle-down", size=10, color="red")))
        fig.update_layout(
            title=f"{ticker} – {model_name} Trade Signals",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            template="plotly_white",
            height=500
        )
        return fig
    else:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="blue")
        buy_signals = df[df["Prediction"] == 1]
        sell_signals = df[df["Prediction"] == -1]
        plt.scatter(buy_signals["Date"], buy_signals["Close"], label="Buy Signal", marker="^", color="green", s=100)
        plt.scatter(sell_signals["Date"], sell_signals["Close"], label="Sell Signal", marker="v", color="red", s=100)
        plt.title(f"{ticker} – {model_name} Trade Signals")
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")
        plt.legend()
        plt.tight_layout()
        return fig

def fetch_ohlcv_data(ticker, start, end):
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Starting OHLCV fetch for {ticker} from {start} to {end}")
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            logger.error(f"No data returned for ticker: {ticker}")
            raise ValueError(f"No data returned for ticker: {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required]
        df['Date'] = pd.to_datetime(df.index)
        df.reset_index(drop=True, inplace=True)
        df = df.dropna()
        if df.columns.duplicated().any():
            logger.warning(f"Duplicate columns found: {df.columns[df.columns.duplicated()].tolist()}")
        logger.info(f"OHLCV data fetched for {ticker}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV data for {ticker}: {str(e)}")
        return pd.DataFrame()

def compute_rsi(series, period=14):
    """
    Computes Relative Strength Index (RSI).

    Args:
        series (pd.Series): Price series.
        period (int): Lookback period.

    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def compute_macd(series, fast=12, slow=26):
    """
    Computes MACD.

    Args:
        series (pd.Series): Price series.
        fast (int): Fast EMA period.
        slow (int): Slow EMA period.

    Returns:
        pd.Series: MACD values.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow).fillna(0)


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """
    Compute Bollinger Bands without zero-filling (avoids y-axis dropping to 0).
    Returns (mid, upper, lower) with NaNs until `window` data points are available.
    """
    s = pd.to_numeric(series, errors="coerce")
    mid = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    '''
    Compute Average Directional Index (ADX).
    '''
    df = pd.DataFrame({'High': high, 'Low': low, 'Close': close}).copy()
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), np.maximum(df['High'] - df['High'].shift(1), 0.0), 0.0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), np.maximum(df['Low'].shift(1) - df['Low'], 0.0), 0.0)
    # Wilder smoothing via rolling sum approximation
    tr_n = df['TR'].rolling(window=period).sum()
    plus_dm_n = df['+DM'].rolling(window=period).sum()
    minus_dm_n = df['-DM'].rolling(window=period).sum()
    plus_di = 100 * (plus_dm_n / tr_n).replace({0: np.nan})
    minus_di = 100 * (minus_dm_n / tr_n).replace({0: np.nan})
    dx = (100 * (abs(plus_di - minus_di) / (plus_di + minus_di))).replace({np.inf: np.nan})
    adx = dx.rolling(window=period).mean()
    return adx.fillna(0)

def compute_pivots(df: pd.DataFrame):
    '''
    Classic floor trader pivots based on previous day's OHLC.
    Adds Pivot, R1-R3, S1-S3 for each row (using prior day's values).
    '''
    piv = pd.DataFrame(index=df.index)
    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    prev_close = df['Close'].shift(1)
    P = (prev_high + prev_low + prev_close) / 3.0
    R1 = 2*P - prev_low
    S1 = 2*P - prev_high
    R2 = P + (prev_high - prev_low)
    S2 = P - (prev_high - prev_low)
    R3 = prev_high + 2*(P - prev_low)
    S3 = prev_low - 2*(prev_high - P)
    piv['Pivot'], piv['R1'], piv['S1'], piv['R2'], piv['S2'], piv['R3'], piv['S3'] = P, R1, S1, R2, S2, R3, S3
    return piv.fillna(0)
def add_technical_indicators(df, include=["MA10", "MA50", "RSI", "MACD", "EMA20", "EMA200", "SMA100", "BBANDS", "ADX", "PIVOTS"]):
    """
    Adds technical indicators to DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        include (list): List of indicators to compute.

    Returns:
        pd.DataFrame: DataFrame with technical indicators.
    """
    df = df.copy()
    if "MA10" in include:
        df["MA10"] = df["Close"].rolling(window=10).mean().fillna(0)
    if "MA50" in include:
        df["MA50"] = df["Close"].rolling(window=50).mean().fillna(0)
    if "RSI" in include:
        df["RSI"] = compute_rsi(df["Close"], period=14)
    if "MACD" in include:
        df["MACD"] = compute_macd(df["Close"])
    # Additional MAs/EMAs
    if "EMA20" in include:
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean().fillna(0)
    if "EMA200" in include:
        df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean().fillna(0)
    if "SMA100" in include:
        df["SMA100"] = df["Close"].rolling(window=100).mean().fillna(0)

    # Bollinger Bands
    if "BBANDS" in include:
        mid, upper, lower = compute_bollinger_bands(df["Close"], window=20, num_std=2)
        df["BB_MID"], df["BB_UPPER"], df["BB_LOWER"] = mid, upper, lower

    # ADX (requires High/Low/Close)
    if "ADX" in include:
        df["ADX"] = compute_adx(df["High"], df["Low"], df["Close"], period=14)

    # Pivot points (based on prior day)
    if "PIVOTS" in include:
        piv = compute_pivots(df)
        df = pd.concat([df, piv], axis=1)

    logging.info(f"Added technical indicators: {include}")
    return df

def create_labels(df, strategy="Return Threshold", threshold=1.0):
    """
    Creates trading signal labels.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' column.
        strategy (str): Labeling strategy ("Simple Up/Down", "Return Threshold", "Price Delta").
        threshold (float): Threshold for returns (%).

    Returns:
        pd.DataFrame: DataFrame with 'Signal' column (-1, 0, 1).
    """
    df = df.copy()
    df["Future_Price"] = df["Close"].shift(-1)
    df["Return"] = ((df["Future_Price"] - df["Close"]) / df["Close"]) * 100

    if strategy == "Simple Up/Down":
        df["Signal"] = np.where(df["Future_Price"] > df["Close"], 1, -1)
    elif strategy == "Return Threshold":
        df["Signal"] = np.where(df["Return"] > threshold, 1,
                                np.where(df["Return"] < -threshold, -1, 0))
    elif strategy == "Price Delta":
        delta = df["Close"].diff().shift(-1)
        df["Signal"] = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    else:
        df["Signal"] = 0

    df.drop(columns=["Future_Price", "Return"], inplace=True)
    logging.info(f"Created labels with strategy: {strategy}, unique signals: {df['Signal'].unique()}")
    return df

def simulate_ml_pnl(df, initial_cash=10000):
    """
    Simulates trading profit and loss based on predictions.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' and 'Prediction' columns.
        initial_cash (float): Initial portfolio cash.

    Returns:
        list: Equity curve.
    """
    df = df.copy()
    position = 0
    cash = initial_cash
    equity_curve = []
    for i in range(len(df) - 1):
        action = df.iloc[i]['Prediction']
        price_today = df.iloc[i]['Close']
        price_next = df.iloc[i + 1]['Close']
        if action == 1 and cash >= price_today:  # Buy
            position += 1
            cash -= price_today
        elif action == -1 and position > 0:  # Sell
            position -= 1
            cash += price_today
        net_worth = cash + position * price_next
        equity_curve.append(net_worth)
    if equity_curve:
        equity_curve.append(cash + position * df.iloc[-1]['Close'])
    logging.info(f"Simulated PnL: final cash={cash}, position={position}")
    return equity_curve

def run_ml_strategy(ticker, start, end, model="Logistic Regression", initial_cash=10000, train_size=0.8, indicators=None):
    """
    Runs the ML strategy pipeline.

    Args:
        ticker (str): Stock ticker symbol.
        start (str or datetime): Start date.
        end (str or datetime): End date.
        model (str): ML model ("Logistic Regression", "Random Forest", "XGBoost").
        initial_cash (float): Initial portfolio cash.
        train_size (float): Fraction of data for training.

    Returns:
        tuple: (DataFrame with predictions, final cash, equity curve).
    """
    try:
        # Fetch and preprocess data
        df = fetch_ohlcv_data(ticker, start, end)
        include_list = indicators if indicators else ["MA10", "MA50", "RSI", "MACD"]
        df = add_technical_indicators(df, include=include_list)
        df = create_labels(df, strategy="Return Threshold", threshold=1.0)
        
        # Validate signals
        valid_signals = {-1, 0, 1}
        if not set(df["Signal"].unique()).issubset(valid_signals):
            logging.error(f"Invalid signals: {df['Signal'].unique()}")
            raise ValueError(f"Invalid signals: {df['Signal'].unique()}")

        # Features
        features = ["MA10", "MA50", "RSI", "MACD"]
        if not all(col in df.columns for col in features):
            logging.error(f"Missing features: {[col for col in features if col not in df.columns]}")
            raise ValueError("Missing required features")

        X = df[features].select_dtypes(include=[np.number]).fillna(0).astype(np.float32)  # cast speeds up XGB
        y_raw = df["Signal"]

        # Label encode for multi-class
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        
        if len(np.unique(y)) < 2:
            logging.error(f"Too few label classes: {np.unique(y)}")
            raise ValueError(f"Too few label classes: {np.unique(y)}")

        # Train-test split
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_size, random_state=42)
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

        # Make X inputs float32 (helps XGBoost speed/memory)
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)

        # Model selection
        model_map = {
            "Logistic Regression": LogisticRegression(multi_class='multinomial', max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(objective='multi:softprob',
                random_state=42,
                n_estimators=300,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                reg_lambda=1.0,
                eval_metric="mlogloss",
                verbosity=0)
        }
        
        clf = model_map.get(model)
        if clf is None:
            logging.error(f"Unknown model: {model}")
            raise ValueError(f"Unknown model: {model}")

        # Train model
        if model == "XGBoost":
            eval_set = [(X_train, y_train), (X_test, y_test)]
            clf.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            clf.fit(X_train, y_train)

        df["Prediction"] = label_encoder.inverse_transform(clf.predict(X))
        
        # Simulate PnL
        equity_curve = simulate_ml_pnl(df, initial_cash)
        final_cash = equity_curve[-1] if equity_curve else initial_cash

        # Select result columns
        result_columns = ['Date', 'Close', 'MA10', 'MA50', 'RSI', 'MACD', 'Signal', 'Prediction']
        result_df = df[result_columns]
        
        logging.info(f"run_ml_strategy completed: df shape={result_df.shape}, final_cash={final_cash}")
        return result_df, final_cash, equity_curve, clf
    except Exception as e:
        logging.error(f"run_ml_strategy failed: {str(e)}")
        raise

def benchmark_models(df):
    """
    Benchmarks ML models and computes performance metrics.

    Args:
        df (pd.DataFrame): DataFrame with features and 'Signal' column.

    Returns:
        pd.DataFrame: DataFrame with model performance metrics.
    """
    results = []
    model_functions = [
        ("Logistic Regression", train_logistic_model),
        ("Random Forest", train_rf_model),
        ("XGBoost", train_xgboost_model)
    ]

    for name, train_func in model_functions:
        try:
            # ✅ Handle 3-return or 4-return gracefully
            returned = train_func(df.copy())
            if len(returned) == 4:
                model, acc, result_df, trained_model = returned
            else:
                model, acc, result_df = returned
                trained_model = model  # fallback for consistency
           
            
            y_true = result_df["Signal"]
            y_pred = result_df["Prediction"]
            
            # Multi-class ROC AUC (one-vs-rest)
            le = LabelEncoder()
            y_true_encoded = le.fit_transform(y_true)
            y_pred_encoded = le.transform(y_pred)
            n_classes = len(le.classes_)
            y_scores = model.predict_proba(result_df[["MA10", "MA50", "RSI", "MACD"]])
            
            roc_auc = []
            for i in range(n_classes):
                try:
                    roc = roc_auc_score((y_true_encoded == i).astype(int), y_scores[:, i])
                    roc_auc.append(roc)
                except:
                    roc_auc.append(0)
            avg_roc_auc = np.mean(roc_auc) if roc_auc else 0

            report = classification_report(y_true_encoded, y_pred_encoded, output_dict=True)
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "ROC AUC": avg_roc_auc,
                "Precision": report["weighted avg"]["precision"],
                "Recall": report["weighted avg"]["recall"],
                "F1-Score": report["weighted avg"]["f1-score"]
            })
        except Exception as e:
            logging.error(f"Benchmark failed for {name}: {str(e)}")
            results.append({
                "Model": name,
                "Accuracy": 0,
                "ROC AUC": 0,
                "Precision": 0,
                "Recall": 0,
                "F1-Score": 0
            })

    return pd.DataFrame(results)

def train_logistic_model(df):
    """
    Trains Logistic Regression model.

    Args:
        df (pd.DataFrame): DataFrame with features and 'Signal' column.

    Returns:
        tuple: (model, accuracy, result DataFrame).
    """
    features = ['MA10', 'MA50', 'RSI', 'MACD']
    X = df[features].select_dtypes(include=[np.number]).fillna(0)
    y = df['Signal'].map({-1: 0, 0: 1, 1: 2})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    df['Prediction'] = model.predict(X)
    df['Prediction'] = df['Prediction'].map({0: -1, 1: 0, 2: 1})
    
    result_df = df[['Date', 'Close', 'MA10', 'MA50', 'RSI', 'MACD', 'Signal', 'Prediction']]
    result_df['model'] = model
    logging.info(f"Logistic Regression: accuracy={acc}, result_df shape={result_df.shape}")
    return model, acc, result_df, model

def train_rf_model(df):
    """
    Trains Random Forest model.

    Args:
        df (pd.DataFrame): DataFrame with features and 'Signal' column.

    Returns:
        tuple: (model, accuracy, result DataFrame).
    """
    features = ['MA10', 'MA50', 'RSI', 'MACD']
    X = df[features].select_dtypes(include=[np.number]).fillna(0)
    y = df['Signal'].map({-1: 0, 0: 1, 1: 2})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    df['Prediction'] = model.predict(X)
    df['Prediction'] = df['Prediction'].map({0: -1, 1: 0, 2: 1})
    
    result_df = df[['Date', 'Close', 'MA10', 'MA50', 'RSI', 'MACD', 'Signal', 'Prediction']]
    result_df['model'] = model 
    logging.info(f"Random Forest: accuracy={acc}, result_df shape={result_df.shape}")
    return model, acc, result_df, model

def train_xgboost_model(df):
    """
    Trains XGBoost model.

    Args:
        df (pd.DataFrame): DataFrame with features and 'Signal' column.

    Returns:
        tuple: (model, accuracy, result DataFrame).
    """
    df = df.copy()
    df = df.dropna()

    features = ['MA10', 'MA50', 'RSI', 'MACD']
    if not all(col in df.columns for col in features):
        logging.error(f"Missing features: {[col for col in features if col not in df.columns]}")
        raise ValueError("Missing required features")

    X = df[features].select_dtypes(include=[np.number]).fillna(0).astype(np.float32)
    y_raw = df["Signal"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    if len(np.unique(y)) < 2:
        logging.error(f"Too few label classes: {np.unique(y)}")
        raise ValueError(f"Too few label classes: {np.unique(y)}")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        reg_lambda=1.0,
        eval_metric="mlogloss",
        verbosity=0
    )

    # Cast to float32 for speed
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    df['Prediction'] = label_encoder.inverse_transform(model.predict(X))
    result_df = df[['Date', 'Close', 'MA10', 'MA50', 'RSI', 'MACD', 'Signal', 'Prediction']]
    result_df['model'] = model 
    logging.info(f"XGBoost: accuracy={acc}, result_df shape={result_df.shape}")
    return model, acc, result_df, model


# Set up logging (use a common log file)
logging.basicConfig(level=logging.INFO, filename="logs/app.log", filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')



# ... [Existing functions like plot_trade_signals, fetch_ohlcv_data, compute_rsi, etc. remain unchanged] ...

def fetch_sentiment_data(symbol, start_date, end_date):
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Starting sentiment fetch for {symbol} from {start_date} to {end_date}")
        feed_url = f"https://news.google.com/rss/search?q={symbol}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(feed_url)
        logger.debug(f"Feed response status: {feed.status}")
        if not feed.entries:
            logger.warning(f"No entries found in feed for {symbol}")
            return pd.DataFrame()
        sentiment_df = pd.DataFrame({
            "Date": [datetime.fromtimestamp(entry.published_parsed) for entry in feed.entries],
            "Title": [entry.title for entry in feed.entries],
            "Sentiment": [0.5] * len(feed.entries)  # Placeholder
        })
        sentiment_df = sentiment_df[(sentiment_df["Date"] >= pd.to_datetime(start_date)) & 
                                   (sentiment_df["Date"] <= pd.to_datetime(end_date))]
        logger.info(f"Sentiment data fetched for {symbol}. Shape: {sentiment_df.shape}")
        return sentiment_df
    except Exception as e:
        logger.error(f"Failed to fetch sentiment data for {symbol}: {str(e)}")
        return pd.DataFrame()
        
        
def compute_risk_metrics(equity: pd.Series, periods_per_year: int = 252, rf: float = 0.0):
    """
    Compute Sharpe, Sortino, and Max Drawdown from a portfolio equity series.
    Returns dict: {'sharpe': float, 'sortino': float, 'max_dd': float}
    """
    import numpy as np
    eq = pd.Series(equity).dropna()
    if eq.empty or len(eq) < 3:
        return {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0}

    rets = eq.pct_change().dropna()
    if rets.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0}

    mu, sigma = rets.mean(), rets.std()
    sharpe = 0.0 if sigma == 0 else ((mu - rf) / sigma) * np.sqrt(periods_per_year)

    downside = rets[rets < 0]
    d_sigma = downside.std()
    sortino = 0.0 if d_sigma == 0 else ((mu - rf) / d_sigma) * np.sqrt(periods_per_year)

    running_max = eq.cummax()
    drawdown = (eq / running_max) - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    return {"sharpe": float(sharpe), "sortino": float(sortino), "max_dd": max_dd}
