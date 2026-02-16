#!/usr/bin/env python3
"""
Title: Linear Regression Predictor
Description: All ML stuff will be kept here.
Date: 2026.02.02
Author: Omar T. Amer
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates the weights for the predictor"
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        help="Gold's generated bandwidth"
    )

    args = parser.parse_args()
    data = pd.read_csv(args.data_csv, header=None)
    values = data.iloc[:, 0].values

    # We need the shape of the data to be
    # (10 samples, the 11th sample)
    # (10 samples, the 11th sample)
    # (10 samples, the 11th sample)
    # ....

    # We do this with a sliding window
    WINDOW = 10

    X = []
    y = []

    for i in range(len(values) - WINDOW):
        inputs = values[i:i + WINDOW]
        baseline = values[i + WINDOW]

        X.append(inputs)
        y.append(baseline)

    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    # Plotting
    time_axis = np.arange(0, len(values), 1, dtype=np.float32)

    preds = []
    for i in range(len(values) - WINDOW):
        window = values[i:i + WINDOW].reshape(1, -1)
        pred = model.predict(window)
        preds.append(pred)


    # Add the first WINDOW points to preds to not mess up
    # the MAE and MSE
    preds = np.array(preds).reshape(-1)
    preds = np.concatenate((values[:WINDOW], preds))
    # breakpoint()
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_axis, values, label="Input")
    ax.plot(time_axis, preds, label="Predicted")
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Bandwidth (Mbps)", fontsize=14)
    ax.set_title("Prediction vs Collected")
    plt.legend()
    plt.savefig("data/pred_" + args.data_csv.name.split('.')[0] + ".svg")

    mae = mean_absolute_error(values, preds)
    mse = mean_squared_error(values, preds)
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Mean Squared Error: {mse:.3f}")

    # Now, store the model
    file_name = "MODEL_" + args.data_csv.name.split('.')[0] + ".pkl"

    with open(f"data/{file_name}", "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to data/{file_name}")