
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
import xgboost as xgb
import optuna

def sw_training(df):
    """
    Train an ensemble model with XGBoost as base models and Random Forest as meta-model using sliding window validation.
    
    Args:
        df (pd.DataFrame): DataFrame containing historical stock prices with a 'close' column for closing prices.
        
    Returns:
        perform (pd.DataFrame): DataFrame containing investment performance metrics and predictions.
    """
    
    # Feature Engineering: Generate time-based and lag features
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week
    df['weekday'] = df.index.dayofweek
    df['lag_1'] = df['close'].shift(1)
    df['lag_2'] = df['close'].shift(2)
    df['lag_3'] = df['close'].shift(3)
    df['rolling_mean_3'] = df['close'].rolling(window=3).mean()
    df['rolling_mean_5'] = df['close'].rolling(window=5).mean()
    df['rolling_mean_7'] = df['close'].rolling(window=7).mean()
    
    # Define target variable (next day's closing price) and prepare initial data split
    last_record = df.iloc[[-1]].reset_index(drop=True)  # Last record for final prediction
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    
    # Initial dataset for model training and testing
    dfp = df.head(1000)
    x = dfp.iloc[:, :-1]  # All columns except 'target'
    y = dfp.iloc[:, -1]   # 'target' column as target variable
    train_size = int(len(dfp) * 0.80)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Define Optuna objective function for hyperparameter tuning
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1.0),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 10),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 10),
            'min_split_loss': trial.suggest_int('min_split_loss', 0, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 0, 20),
            'objective': 'reg:squarederror'
        }
        reg = xgb.XGBRegressor(**params)
        reg.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        y_pred = reg.predict(x_test)
        return mean_squared_error(y_test, y_pred)
    
    # Hyperparameter optimization for each base model using Optuna
    params = []
    for _ in range(10):  # Tune parameters for 10 base models
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        params.append(study.best_params)

    # Define function to train each XGBoost model with specified parameters
    def train_xgboost_model(X_train, y_train, param):
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train)
        return model

    # Initialize sliding window variables
    j, exp, invest = 1000, 0, 1000
    perform = pd.DataFrame(columns=['current', 'future', 'predicted', 'return', 'investment', 'diff', 
                                    'resulte', 'actual dir', 'predicted dir'], 
                           index=range(len(df) - j))

    # Sliding window validation loop
    for i in range(len(df) - j):
        # Update training and test datasets in each iteration
        dff = df.iloc[i:j, :]
        split_index = int(len(dff) * 0.8)
        X_train, Y_train = dff.iloc[:split_index, :-1], dff.iloc[:split_index, -1]
        X_test, Y_test = dff.iloc[split_index:, :-1], dff.iloc[split_index:, -1]
        
        # Train base models in parallel and collect predictions
        models = Parallel(n_jobs=-1)(delayed(train_xgboost_model)(X_train, Y_train, param) for param in params)
        train_meta_features = np.column_stack([model.predict(X_train) for model in models])
        test_meta_features = np.column_stack([model.predict(X_test) for model in models])
        
        # Train Random Forest meta-model
        meta_model = RandomForestRegressor(n_estimators=100, max_depth=3)
        meta_model.fit(train_meta_features, Y_train)
        
        # Make final predictions using the meta-model
        test_final_predictions = meta_model.predict(test_meta_features)
        
        # Join predictions with actual values for performance metrics
        predictions = pd.DataFrame(test_final_predictions, index=Y_test.index, columns=['predicted'])
        Y_test = Y_test.to_frame().join(predictions)
        
        # Collect metrics in `perform` DataFrame
        perform.loc[exp, 'current'] = X_train.iloc[-1, 1]
        perform.loc[exp, 'future'] = Y_test.iloc[0, 0]
        perform.loc[exp, 'predicted'] = Y_test.iloc[0, 1]
        
        # Calculate returns and update investment based on prediction accuracy
        if ((perform.loc[exp, 'future'] - perform.loc[exp, 'current']) * 
            (perform.loc[exp, 'predicted'] - perform.loc[exp, 'current'])) > 0:
            perform.loc[exp, 'return'] = abs((perform.loc[exp, 'future'] - perform.loc[exp, 'current']) / perform.loc[exp, 'current'])
            invest *= (1 + perform.loc[exp, 'return'])
            perform.loc[exp, 'investment'] = invest
            perform.loc[exp, 'resulte'] = 1
            perform.loc[exp, 'actual dir'] = int(perform.loc[exp, 'future'] > perform.loc[exp, 'current'])
            perform.loc[exp, 'predicted dir'] = int(perform.loc[exp, 'predicted'] > perform.loc[exp, 'current'])
        else:
            perform.loc[exp, 'return'] = -abs((perform.loc[exp, 'future'] - perform.loc[exp, 'current']) / perform.loc[exp, 'current'])
            invest *= (1 + perform.loc[exp, 'return'])
            perform.loc[exp, 'investment'] = invest
            perform.loc[exp, 'resulte'] = 0
            perform.loc[exp, 'actual dir'] = int(perform.loc[exp, 'future'] > perform.loc[exp, 'current'])
            perform.loc[exp, 'predicted dir'] = int(perform.loc[exp, 'predicted'] > perform.loc[exp, 'current'])
            
        perform.loc[exp, 'diff'] = perform.loc[exp, 'future'] - perform.loc[exp, 'predicted']
        exp += 1
        j += 1

    # Clean up performance DataFrame by removing NaN values
    perform = perform.apply(pd.to_numeric, errors='coerce')
    perform.dropna(inplace=True)
    
    # Display performance metrics
    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy_score(perform['actual dir'], perform['predicted dir'])}")
    print(f"Precision: {precision_score(perform['actual dir'], perform['predicted dir'])}")
    print(f"Recall: {recall_score(perform['actual dir'], perform['predicted dir'])}")
    print(f"F1 Score: {f1_score(perform['actual dir'], perform['predicted dir'])}")
    
    # Generate prediction for the latest record
    latest_prediction = meta_model.predict(np.column_stack([model.predict(last_record) for model in models]))
    current_price = last_record.close.values[0]
    price_change = (latest_prediction - current_price) * 100 / current_price
    print('********************************************************************************************')
    if price_change > 0:
        print(f'The current price is {current_price} and the Predicted price is {latest_prediction[0]}')
        print(f'There is a buy opportunity with an expected price change of {price_change[0]:.3f}%')
    elif price_change < 0:
        print(f'The current price is {current_price} and the Predicted price is {latest_prediction[0]}')
        print(f'There is a sell opportunity with an expected price change of {price_change[0]:.3f}%')
    else:
        print('No significant expected volatility for today.')
    print('********************************************************************************************')
    
    return perform


def sw_analyse_performance(perform):
    """
    Analyze and visualize the performance of the sliding window ensemble model strategy.
    
    Args:
        perform (pd.DataFrame): DataFrame containing investment performance metrics, including actual and predicted values.
        
    Returns:
        dict: A dictionary of key performance metrics including win rate, MSE, expected daily return, maximum loss,
              estimated monthly return, and monthly Sharpe ratio.
    """
    
    # 1. Calculate Key Performance Metrics
    winrate = sum(perform["resulte"]) / len(perform)  # Win rate of the investment strategy
    perform['SE'] = (perform['future'] - perform['predicted']) ** 2  # Squared error for MSE calculation
    mse = perform['SE'].mean()  # Mean Squared Error
    dr = perform['return'].mean()  # Average Daily Return
    maxreturn = perform['return'].max()  # Maximum gain in a single trade
    maxloss = perform['return'].min()  # Maximum loss in a single trade
    
    # Monthly Return and Sharpe Ratio (assuming 21 trading days in a month)
    mr = (1 + dr) ** 21 - 1  # Estimated monthly return
    dsr = dr / perform['return'].std()  # Daily Sharpe Ratio
    msr = dsr * np.sqrt(21)  # Monthly Sharpe Ratio

    # Print Metrics Summary
    print(75 * '*')
    print(f'Win Rate: {round(winrate * 100, 1)}%')
    print(f'MSE: {mse}')
    print(f'Expected Daily Return: {round(dr * 100, 1)}%')
    print(f'Maximum Gain: {round(maxreturn * 100, 1)}%')
    print(f'Maximum Loss: {round(maxloss * 100, 1)}%')
    print(f'Estimated Monthly Return: {round(mr * 100, 1)}%')
    print(f'Monthly Sharpe Ratio: {round(msr, 2)}')
    print(75 * '*')
    
    # 2. Visualization of Performance Metrics

    # Plot: Actual vs. Predicted Prices Over the Last Year (Daily Timeframe)
    plt.figure(figsize=(12, 6))
    plt.plot(perform.index[-252:], perform['future'].iloc[-252:], label='Actual Prices', color='blue')
    plt.scatter(perform.index[-252:], perform['predicted'].iloc[-252:], label='Predicted Prices', color='red', marker='*')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.title('Actual and Predicted Prices Over Last Year (D1 Timeframe)')
    plt.legend()
    plt.show()

    # Plot: Capital Growth Over the Last Year
    plt.figure(figsize=(12, 6))
    plt.plot(perform.index[-252:], perform['investment'].iloc[-252:], label='Accumulated Investment', color='green')
    plt.xlabel('Time')
    plt.ylabel('Investment Value')
    plt.title('Capital Growth in the Last Year')
    plt.legend()
    plt.show()

    # Plot: Return Distribution with Mean and Quantiles
    mean = perform['return'].mean()
    q05 = perform['return'].quantile(0.05)
    q95 = perform['return'].quantile(0.95)
    
    plt.figure(figsize=(12, 6))
    sns.histplot(perform['return'], bins=40, kde=True, color='blue', edgecolor='black')
    plt.axvline(mean, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(q05, color='red', linestyle='--', linewidth=2, label=f'0.05 Quantile: {q05:.2f}')
    plt.axvline(q95, color='red', linestyle='--', linewidth=2, label=f'0.95 Quantile: {q95:.2f}')
    plt.title('Return Distribution with Mean and Quantiles')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Plot: Q-Q Plot of Residuals to Assess Normality
    plt.figure(figsize=(10, 6))
    stats.probplot(perform['diff'], dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.show()

    # Plot: Autocorrelation of Residuals
    plt.figure(figsize=(10, 6))
    plot_acf(perform['diff'], lags=30)
    plt.title('Autocorrelation of Residuals')
    plt.show()

    # 3. Return Key Performance Metrics
    return {
        'win_rate': round(winrate * 100, 1),
        'mse': mse,
        'expected_daily_return': round(dr * 100, 1),
        'maximum_gain': round(maxreturn * 100, 1),
        'maximum_loss': round(maxloss * 100, 1),
        'estimated_monthly_return': round(mr * 100, 1),
        'monthly_sharpe_ratio': round(msr, 2)
    }







