# -*- coding: utf-8 -*-

# Preparation for running locally:
#   pip install kaggle numpy pandas tensorflow transformers
#   mkdir -p ~/.kaggle
#   cp kaggle.json ~/.kaggle/
#   ls ~/.kaggle
#   chmod 600 /root/.kaggle/kaggle.json
#   kaggle competitions download -c tabular-playground-series-oct-2021 -p input

import math
import os
import sys
import time
import numpy as np
import pandas as pd
import shutil

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

nrows_limit = None  # 16000
batch_size = None  # 100_000
train_ratio = .80
max_epochs = 1000
max_epochs_with_worse_val_loss = 10
learning_rate = .0015

cols = 285


def read(fname, labeled, nrows=None):
    dtype = {"id": str}
    for i in range(cols):
        dtype[f"f{i}"] = np.single
    if labeled:
        dtype["target"] = np.single
    p = os.path.join(os.pardir, "input", fname)
    return pd.read_csv(p, dtype=dtype, nrows=nrows)


def solve(timestamp, train_x, train_y, val_x, val_y, test_x):
    log_dir = os.path.join(os.pardir, "logs", timestamp)
    min_val_loss = math.inf
    epochs_with_worse_val_loss = 0

    def settle_down(epoch, logs):
        nonlocal min_val_loss, epochs_with_worse_val_loss
        val_loss = logs["val_loss"]
        if min_val_loss >= val_loss:
            min_val_loss = val_loss
            epochs_with_worse_val_loss = 0
        else:
            epochs_with_worse_val_loss += 1
            if epochs_with_worse_val_loss > max_epochs_with_worse_val_loss:
                model.stop_training = True

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cols, ), dtype=tf.float32),
        tf.keras.layers.Dense(81, activation="linear", use_bias=True),
        tf.keras.layers.Dense(9, activation="hard_sigmoid", use_bias=False),
        tf.keras.layers.Dense(1, activation="sigmoid", use_bias=False),
    ])
    model.summary()
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss=loss, optimizer=optimizer)
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
    if val_x is None:
        validation_data = None
    else:
        validation_data = (val_x, val_y)
        callbacks += [
            tf.keras.callbacks.LambdaCallback(on_epoch_end=settle_down),
        ]
    model.fit(x=train_x,
              y=train_y,
              epochs=max_epochs,
              batch_size=batch_size,
              callbacks=callbacks,
              validation_data=validation_data,
              verbose=2)
    return model.predict(test_x)


def main():
    timestamp = time.strftime("%Y%m%d-%H%M")
    shutil.copyfile(sys.argv[0], timestamp + ".py")

    labeled_df = read("train.csv", labeled=True, nrows=nrows_limit)
    test_df = read("test.csv", labeled=False, nrows=nrows_limit)

    train_size = int(len(labeled_df) * train_ratio)
    np.random.seed(int(19680516 * train_ratio))
    labeled_pick = np.random.permutation(labeled_df.index)
    train_df = labeled_df.iloc[labeled_pick[:train_size]]
    val_df = labeled_df.iloc[labeled_pick[train_size:]]
    print(len(train_df), "samples to train on")
    print(len(val_df), "samples to validate")
    print(len(test_df), "samples to test")

    train_x = pd.concat([train_df[f"f{i}"] for i in range(cols)], axis=1)
    val_x = pd.concat([val_df[f"f{i}"] for i in range(cols)],
                      axis=1) if train_ratio < 1 else None
    test_x = pd.concat([test_df[f"f{i}"] for i in range(cols)], axis=1)
    train_y = train_df["target"]
    val_y = val_df["target"]

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        test_y = solve(timestamp=timestamp,
                       train_x=train_x,
                       train_y=train_y,
                       val_x=val_x,
                       val_y=val_y,
                       test_x=test_x)

    submission = test_df[["id"]].assign(target=test_y)
    submission.to_csv("submission.csv", index=False)


main()
