import os
from operator import index

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import re


keras.config.disable_traceback_filtering()
print(tf.__version__)
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.list_logical_devices('GPU')

import matplotlib.pyplot as plt
# ---------------------------------------------------------
# 1. Load labels (your CSV) and voxel volume
# ---------------------------------------------------------

VOXEL_SIZE = 128
filler_info = "/home/rc/pythonProject_img23d/data/out_microstructures/csv"
npz_root = "/home/rc/pythonProject_img23d/data/vox_npy"
type_names = ["sphere", "cube", "cylinder", "ellipsoid"]
mapping ={"sphere":0, "cube":1, "cylinder":2, "ellipsoid":3}

def load_data(npz_root, filler_info, start=0, stop=100, step=10):
    filler_info_files = sorted(glob.glob(filler_info+"/*.csv"))
    print("file number:", len(filler_info_files))
    y_center_list = []
    y_size_list = []
    y_vec_list = []
    y_type_names_list = []
    for f in filler_info_files[start:stop:step]:
        df = pd.read_csv(f)
        y_center = df[["center_x", "center_y", "center_z"]].values.astype("float32")
        y_size = df[["s1", "s2", "s3"]].values.astype("float32")
        y_vec = df[["vec_x", "vec_y", "vec_z"]].values.astype("float32")
        type_names = df["type"]
        type_names_encoded = np.vectorize(mapping.get)(type_names)

        y_center_list.append(y_center)
        y_size_list.append(y_size)
        y_vec_list.append(y_vec)
        y_type_names_list.append(type_names_encoded)

    y_center_list_combined = np.concatenate(y_center_list, axis=0)
    y_size_list_combined = np.concatenate(y_size_list, axis=0)
    y_vec_list_combined = np.concatenate(y_vec_list, axis=0)
    y_type_names_list_combined = np.concatenate(y_type_names_list, axis=0)
    uni, eco = np.unique(y_type_names_list_combined, return_counts=True)
    print("len_name", len(y_type_names_list_combined))
    print(uni)
    print(eco)

    npz_root_files = sorted(glob.glob(npz_root+"/*.npz"))
    print("file number npz:", len(npz_root_files))
    x_volume = []
    for f in npz_root_files[start:stop:step]:
        with np.load(f, allow_pickle=False) as z:
            volume = z["vol"]
            volume_N = np.zeros([np.max(volume), VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])
            for i in range(np.max(volume)):
                volume_N[i][volume == i + 1] = 1

        x_volume.append(volume_N.astype(dtype=np.float32))
    x_volume_combined = np.concatenate(x_volume, axis=0)
    return np.expand_dims(x_volume_combined, axis=-1).astype(dtype=np.int8), {
                            "type":  y_type_names_list_combined,
                            "center": y_center_list_combined,
                            "size":   y_size_list_combined,
                            "vec":    (y_vec_list_combined+1)/2,
                        }


X, Y = load_data(npz_root, filler_info, start=0, stop=150000, step=1)
print(X.shape)
print(np.max(X))
# dataset = tf.data.Dataset.from_tensor_slices((X, Y))
# dataset = dataset.shuffle(10).batch(16).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------
# 3. Define a multi-task 3D CNN model
# ---------------------------------------------------------
inputs = keras.Input(shape=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1))

x = layers.Conv3D(16, 5, padding="same", activation="relu")(inputs)
x = layers.MaxPool3D(2)(x)
x = layers.Conv3D(32, 5, padding="same", activation="relu")(x)
x = layers.MaxPool3D(2)(x)
x = layers.Conv3D(64, 5, padding="same", activation="relu")(x)
x = layers.MaxPool3D(2)(x)
x = layers.Conv3D(128, 5, padding="same", activation="relu")(x)
x = layers.MaxPool3D(2)(x)
x = layers.Conv3D(256, 5, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling3D()(x)
x = layers.Dense(128, activation="relu")(x)

# Outputs
type_out   = layers.Dense(len(type_names), activation="softmax", name="type")(x)
center_out = layers.Dense(3, name="center")(x)
size_out   = layers.Dense(3, name="size")(x)
vec_out    = layers.Dense(3, name="vec")(x)

model = keras.Model(inputs, [type_out, center_out, size_out, vec_out])

model.compile(
    optimizer=keras.optimizers.Adam(2e-4),
    loss={
        "type": "sparse_categorical_crossentropy",
        "center": "mae",
        "size": "mae",
        "vec": "mae",
    },
    loss_weights={"type": 1.0, "center": 1.0, "size": 1.0, "vec": 1.0},
    metrics={"type": "accuracy"},
)

model.summary()

# ---------------------------------------------------------
# 4. Train
# ---------------------------------------------------------

history = model.fit(
    X,Y,
    batch_size=16,
    epochs=100,
    validation_split=0.1,
    shuffle=False,
)
ckpt = tf.train.Checkpoint(model=model)
os.makedirs("result/ckpt", exist_ok=True)
os.makedirs("result/pred_csv", exist_ok=True)
ckpt_manager = tf.train.CheckpointManager(ckpt, "result/ckpt/", max_to_keep=2)
ckpt_manager.save()

# Convert to DataFrame
hist_df = pd.DataFrame(history.history)
# Save to CSV
hist_df.to_csv("result/training_history.csv", index=True)

# prediction
predict_root = "/home/rc/pythonProject_img23d/ident/vox_npy"
predict_root = sorted(glob.glob(predict_root + "/*.npz"))
index = 0
for f in predict_root:
    with np.load(f, allow_pickle=False) as z:
        volume = z["vol"]
        volume_N = np.zeros([np.max(volume), VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])
        for i in range(np.max(volume)):
            volume_N[i][volume == i + 1] = 1

    pre_type, pre_center, pre_size, pre_vec = model.predict(volume_N)
    pre_vec = pre_vec*2-1
    type_id = np.argmax(pre_type, axis=-1)

    type_name = []
    for i in range(len(type_id)):
        type_name.append(type_names[type_id[i]])

    row = {"composite_id": np.zeros(len(volume_N)),
           "filler_index": range(len(volume_N)),
           "type":type_name,
           "center_x":pre_center[:,0],
           "center_y":pre_center[:,1],
           "center_z":pre_center[:,2],
           "s1":pre_size[:,0],
           "s2":pre_size[:,1],
           "s3":pre_size[:,2],
           "vec_x":pre_vec[:,0],
           "vec_y":pre_vec[:,1],
           "vec_z":pre_vec[:,2]
    }
    df_pred = pd.DataFrame(row)
    df_pred.to_csv("result/pred_csv/pre" + str(index) + ".csv", index=False)
    index += 1