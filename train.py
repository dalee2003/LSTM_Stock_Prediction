import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


# Path to unzipped datasets
dataset_path = '/Users/daphnelee/stock_test/'

# List all CSV files in the dataset directory
stock_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

def create_model(x_train, lstm_units=50, dense_rate=160):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dense(dense_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_select_best_model(x_train, y_train, x_test, y_test, scaler,  lstm_units=50, dense_rate=160, num_trials=5, rmse_threshold=50):
    best_model = None
    best_val_loss = np.inf
    best_rmse = np.inf

    for i in range(num_trials):
        print(f"Trial {i+1}/{num_trials}")
        curr_model = create_model(x_train, lstm_units, dense_rate)
        history = curr_model.fit(x_train, y_train, batch_size=1, epochs=3, validation_split=0.2, verbose=0)

        # Get the validation loss of the current model
        val_loss = history.history['val_loss'][-1]

        # Make predictions
        curr_model_pred = curr_model.predict(x_test)
        
        # Inverse transform the predictions
        #curr_model_pred_reshaped = np.zeros((curr_model_pred.shape[0], 7))
        curr_model_pred_reshaped = np.zeros((curr_model_pred.shape[0], x_test.shape[2]))

        curr_model_pred_reshaped[:, 0] = curr_model_pred[:, 0]  # Only fill the 'Open' price predictions

        # Ensure that only the first feature is used for RMSE calculation
        curr_pred = scaler.inverse_transform(curr_model_pred_reshaped)[:, 0]  # Extract 'Open' price predictions

        #val_rmse = np.sqrt(np.mean((curr_pred - scaler.inverse_transform(x_test)[:, 0])**2))
        val_rmse = np.sqrt(np.mean((curr_pred - y_test)**2))

        # Check if this model has the best validation RMSE so far
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = curr_model
            best_val_loss = val_loss

        print(f"Validation RMSE for trial {i+1}: {val_rmse}")

    return best_model, best_val_loss, best_rmse

def train_and_save_model(stock_data, stock_name):
    stock_data['MA50'] = stock_data['Open'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Open'].rolling(window=200).mean()

    # Drop NaN values created by moving averages
    stock_data = stock_data.dropna()

    # Filter the relevant columns
    data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200']]
    dataset = data.values

    # Split the data into training and testing datasets
    training_data_len = math.ceil(len(dataset) * 0.8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training dataset
    train_data = scaled_data[0:training_data_len, :]
    test_data = scaled_data[training_data_len-60:, :]

    # Split the data into x_train and y_train data set
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, :])
        y_train.append(train_data[i, 0])

    x_test, y_test = [], []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, :])
        y_test.append(test_data[i, 0])

    # Convert the x_train, y_train, x_test, y_test to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Reshape the data into 3 dimensional
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Hyperparameters to try if RMSE is not small enough
    lstm_units_list = [50, 60, 100]
    dense_rate_list = [160, 150, 170]
    epochs_list = [3, 5]
    best_rmse = np.inf

    # Train and select the best model
    #best_model, best_val_loss, best_rmse = train_and_select_best_model(x_train, y_train, x_test, y_test)

    # If the best RMSE is not small enough, alter parameters and run the trials again
    if best_rmse > 50:
        for lstm_units in lstm_units_list:
            for dense_rate in dense_rate_list:
                for epochs in epochs_list:
                    print(f"Trying with LSTM units: {lstm_units}, Dense rate: {dense_rate}, Epochs: {epochs}")

                    best_model, best_val_loss, best_rmse = train_and_select_best_model(x_train, y_train, x_test, y_test,scaler, lstm_units, dense_rate, num_trials=5)

                    if best_rmse <= 50:
                        break
                if best_rmse <= 50:
                    break
            if best_rmse <= 50:
                break

    print("Best validation loss:", best_val_loss)
    print("Best validation RMSE:", best_rmse)

    # Save the best model to the 'models' directory
    model_directory = 'models'
    best_model.save(os.path.join(model_directory, f'{stock_name}.h5'))

# Process each stock file
for stock_file in stock_files:
    stock_name = os.path.splitext(stock_file)[0]
    stock_data = pd.read_csv(os.path.join(dataset_path, stock_file))

    if 'Close' in stock_data.columns:
        train_and_save_model(stock_data, stock_name)
    else:
        print(f"Skipping {stock_file} as it does not contain 'Close' column.")

print("All models trained and saved.")

