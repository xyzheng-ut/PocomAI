import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras import ops
import trimesh
import glob
import pandas as pd


print(tf.__version__)
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.list_logical_devices('GPU')


def create_dir(path: str):
    """
    Create a directory of it does not exist
    :param path: Path to directory
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


create_dir("./result/npz")
create_dir("./result/ply")
create_dir("./result/ckpt")



def hsv_to_rgb(h, s, v):
    """h,s,v in [0,1] -> r,g,b in [0,1]"""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else:        r, g, b = v, p, q
    return r, g, b


def label_to_rgb(label):
    """
    Deterministic, well-separated color per integer label (>=1).
    Uses golden-ratio hue spacing + mild lightness jitter for variety.
    """
    phi = 0.6180339887498949
    h = (label * phi) % 1.0
    s = 0.65 + 0.25 * (((label * 37) % 7) / 6.0)      # vary saturation slightly
    v = 0.85 - 0.10 * (((label * 17) % 5) / 4.0)      # vary value slightly
    r, g, b = hsv_to_rgb(h, min(s, 0.95), max(min(v, 0.95), 0.6))
    return (int(r * 255), int(g * 255), int(b * 255))


# ---------- Box boundary (NEW) ----------

def make_cylinder(radius, length, axis, sections=30):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    axis = np.array([0., 0., 1.]) if n < 1e-12 else axis / n
    m = trimesh.creation.cylinder(radius=float(radius), height=float(length), sections=sections)
    R = trimesh.geometry.align_vectors(np.array([0., 0., 1.]), axis)
    m.apply_transform(R)
    return m


def translate_mesh(mesh, center):
    T = np.eye(4); T[:3, 3] = np.asarray(center, float)
    mesh = mesh.copy(); mesh.apply_transform(T); return mesh


def apply_color(mesh, rgba):
    rgba = np.asarray(rgba, dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
    return mesh


def make_box_wireframe(size=1.0, edge_radius=0.003, color=[40, 40, 40, 255] ):
    """
    12 edges as thin cylinders along the unit box [0,size]^3.
    """
    s = float(size)
    # 8 corners
    C = np.array([[0,0,0],[s,0,0],[0,s,0],[s,s,0],[0,0,s],[s,0,s],[0,s,s],[s,s,s]], dtype=float)
    # 12 edges as (start_idx, end_idx)
    edges = [
        (0,1),(2,3),(4,5),(6,7),  # x edges
        (0,2),(1,3),(4,6),(5,7),  # y edges
        (0,4),(1,5),(2,6),(3,7),  # z edges
    ]
    tubes = []
    for a,b in edges:
        p1, p2 = C[a], C[b]
        axis = p2 - p1
        length = np.linalg.norm(axis)
        if length <= 0:  # guard
            continue
        m = make_cylinder(edge_radius, length, axis)
        mid = (p1 + p2) * 0.5
        m = translate_mesh(m, mid)
        tubes.append(apply_color(m, color))
    if len(tubes) == 1:
        return tubes[0]
    return trimesh.util.concatenate(tubes)


def export_cubes_ply(vol, res, rve_size, out_path, max_cubes=int(3e6)):
    """
    Export non-zero voxels as a colored **mesh** of cubes (12 triangles per voxel).
    Automatically falls back to points if too many voxels.
    """
    idx = np.where(vol > 0)
    nvox = int(idx[0].size)
    if nvox == 0:
        # write empty mesh
        trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3), dtype=np.int64)).export(out_path)
        return
    if nvox > max_cubes:
        # too big: fallback to points
        print(f"  [info] {out_path.name}: {nvox} voxels > max_cubes={max_cubes}; exporting points instead.")
        # build grid centers again for this quick path
        s = rve_size / res
        Xc = (idx[1].astype(np.float64) + 0.5) * s
        Yc = (idx[0].astype(np.float64) + 0.5) * s
        Zc = (idx[2].astype(np.float64) + 0.5) * s
        V = np.stack([Xc, Yc, Zc], axis=1)
        labels = vol[idx].astype(np.int64)
        lut = np.zeros((labels.max() + 1, 3), dtype=np.uint8)
        for lab in np.unique(labels):
            if lab <= 0: continue
            lut[lab] = np.array(label_to_rgb(int(lab)), dtype=np.uint8)
        colors = lut[labels]
        trimesh.PointCloud(vertices=V, colors=colors).export(out_path)
        return

    # Build cubes per label to avoid mixing colors
    s = rve_size / res
    half = 0.5 * s
    # 8 corners of a unit cube centered at origin (scaled by s)
    cube_verts = np.array([
        [-half, -half, -half],
        [ half, -half, -half],
        [ half,  half, -half],
        [-half,  half, -half],
        [-half, -half,  half],
        [ half, -half,  half],
        [ half,  half,  half],
        [-half,  half,  half],
    ], dtype=np.float64)
    # 12 triangles (two per face), using the above vertex order
    cube_faces = np.array([
        [0,1,2], [0,2,3],   # z-
        [4,5,6], [4,6,7],   # z+
        [0,1,5], [0,5,4],   # y-
        [3,2,6], [3,6,7],   # y+
        [0,3,7], [0,7,4],   # x-
        [1,2,6], [1,6,5],   # x+
    ], dtype=np.int64)

    labels_all = vol[idx].astype(np.int64)
    unique_labels = np.array(sorted(list(set(labels_all.tolist()))), dtype=np.int64)

    verts_all = []
    faces_all = []
    colors_all = []

    v_offset = 0
    for lab in unique_labels:
        mask_lab = labels_all == lab
        iy, ix, iz = idx[0][mask_lab], idx[1][mask_lab], idx[2][mask_lab]
        n = int(iy.size)
        if n == 0:
            continue

        # centers in world coords (note axis mapping)
        centers = np.stack([
            (ix.astype(np.float64) + 0.5) * s,
            (iy.astype(np.float64) + 0.5) * s,
            (iz.astype(np.float64) + 0.5) * s
        ], axis=1)  # (n,3)

        # replicate cube template
        verts = (centers[:, None, :] + cube_verts[None, :, :]).reshape(-1, 3)   # (n*8, 3)

        # faces with offset per cube
        f = (cube_faces[None, :, :] + (np.arange(n)[:, None, None] * 8)).reshape(-1, 3)
        faces_all.append(f + v_offset)

        # per-vertex colors for this label
        rgb = np.array(label_to_rgb(int(lab)), dtype=np.uint8)
        colors = np.tile(rgb[None, :], (n * 8, 1))
        colors_all.append(colors)

        verts_all.append(verts)
        v_offset += n * 8

    if len(verts_all) == 0:
        trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3), dtype=np.int64)).export(out_path)
        return

    V = np.vstack(verts_all)
    F = np.vstack(faces_all)
    C = np.vstack(colors_all)

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    # attach per-vertex colors (PLY supports vertex colors)
    mesh.visual.vertex_colors = C

    # add framework
    geoms = []
    geoms.append(mesh)

    box = make_box_wireframe(size=rve_size, edge_radius=0.003)
    geoms.append(box)
    # Build a scene and export
    scene = trimesh.Scene()
    for i, m in enumerate(geoms):
        scene.add_geometry(m, node_name=m.metadata.get("name", f"mesh_{i}") if hasattr(m, "metadata") else f"mesh_{i}")

    merged = trimesh.util.concatenate([g for g in geoms if isinstance(g, trimesh.Trimesh)])

    merged.export(out_path)


def load_data(npz_root, VOXEL_SIZE=64, start=0, stop=100, step=10):

    npz_root_files = sorted(glob.glob(npz_root+"/*.npz"))
    npz_root_files = npz_root_files[start:stop:step]
    volume_N = np.zeros((len(npz_root_files), VOXEL_SIZE,VOXEL_SIZE,VOXEL_SIZE))
    for i in range(len(npz_root_files)):
        with np.load(npz_root_files[i], allow_pickle=False) as z:
            volume = z["vol"]
            volume[volume>1] = 1
        volume_N[i,:,:,:] = volume

    return tf.expand_dims(volume_N, axis=-1)


def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=1)[0]
    return psnr_value


class EDSRModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, x):
        # Adding dummy dimension using tf.expand_dims and converting to float32 using tf.cast
        # x = ops.cast(tf.expand_dims(x, axis=0), dtype="float32")
        # Passing low resolution image to model
        super_resolution_img = self(x, training=False)
        # Clips the tensor from min(0) to max(255)
        super_resolution_img = ops.clip(super_resolution_img, 0, 1)
        # Rounds the values of a tensor to the nearest integer
        super_resolution_img = ops.round(super_resolution_img)
        # Removes dimensions of size 1 from the shape of a tensor and converting to uint8
        # super_resolution_img = ops.squeeze(
        #     ops.cast(super_resolution_img, dtype="uint8"), axis=0
        # )
        return super_resolution_img


# Residual Block
def ResBlock(inputs):
    x = layers.Conv3D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv3D(64, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x



def depth_to_space_3d(x, block_size):
    # x: [batch, D, H, W, C]
    bsize = block_size
    batch_size, d, h, w, c = tf.unstack(tf.shape(x))
    c //= (bsize ** 3)
    x = tf.reshape(
        x,
        (batch_size, d, h, w, bsize, bsize, bsize, c)
    )
    # reorder axes to move spatial sub-blocks into depth, height, and width
    x = tf.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7))
    # combine expanded spatial dimensions
    x = tf.reshape(x, (batch_size, d * bsize, h * bsize, w * bsize, c))
    return x


# Upsampling Block
def Upsampling(inputs, factor=2, **kwargs):
    x = layers.Conv3D(64 * (factor**3), 3, padding="same", **kwargs)(inputs)
    x = layers.Lambda(lambda x: depth_to_space_3d(x, block_size=factor))(x)
    # x = layers.Conv3D(64 * (factor**3), 3, padding="same", **kwargs)(x)
    # x = layers.Lambda(lambda x: depth_to_space_3d(x, block_size=factor))(x)
    return x


def make_model(num_filters, num_of_residual_blocks):
    # Flexible Inputs to input_layer
    input_layer = layers.Input(shape=(None, None, None, 1))
    # Scaling Pixel Values
    # x = layers.Rescaling(scale=1.0 / 1.)(input_layer)
    x = x_new = layers.Conv3D(num_filters, 3, padding="same")(input_layer)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock(x_new)

    x_new = layers.Conv3D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])

    x = Upsampling(x)
    output_layer = layers.Conv3D(1, 3, padding="same")(x)

    # output_layer = layers.Rescaling(scale=1)(x)
    return EDSRModel(input_layer, output_layer)


model = make_model(num_filters=64, num_of_residual_blocks=8)
model.summary()
# Using adam optimizer with initial learning rate as 1e-4, changing learning rate after 5000 steps to 5e-5
optim_edsr = keras.optimizers.Adam(
    learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-3, 1e-4]
    )
)
# Compiling model with loss as mean absolute error(L1 Loss) and metric as psnr
model.compile(optimizer=optim_edsr, loss="mae", metrics=[PSNR])
# Training for more epochs will improve results
predict_root = "/home/rc/pythonProject_img23d/ident/vox_npy64"
predict_root = sorted(glob.glob(predict_root + "/*.npz"))
volume_N = np.zeros([5, 64, 64, 64])
for i in range(5):
    f = predict_root[-i-1]
    with np.load(f, allow_pickle=False) as z:
        volume = z["vol"]
        volume[volume > 1] = 1
        volume_N[i, :, :, :] = volume
# volume_N = tf.cast(volume_N, tf.float16)
volume_N = tf.expand_dims(volume_N, axis=-1)
print(volume_N.shape)

class PlotCallback(keras.callbacks.Callback):
    def __init__(self, data):
        super().__init__()
        self.volume_N = data
    def on_epoch_end(self,epoch, log={}):
        generated_voxels = self.model.predict(volume_N)
        print(generated_voxels.shape)
        # generated_voxels = np.round(np.clip(generated_voxels, 0, 1)).astype(np.bool_)
        for i in range(generated_voxels.shape[0]):
            export_cubes_ply(generated_voxels[i], 128, 1, f"./result/ply/generated_{epoch}_{i}.ply",
                             max_cubes=128**3)
        # plot_mul_3D_voxels(generated_voxels, save_name="./training_results/img/generated_%d" % e)
        np.savez_compressed(f"./result/npz/generated_{epoch}", vol=generated_voxels)


dataset_x = load_data("/home/rc/pythonProject_img23d/data/vox_npy64", VOXEL_SIZE=64, start=0, stop=150000, step=1)  # [N,64,64,64,1]
dataset_y = load_data("/home/rc/pythonProject_img23d/data/vox_npy128", VOXEL_SIZE=128, start=0, stop=150000, step=1)
dataset_x = tf.cast(dataset_x, tf.float16)
dataset_y = tf.cast(dataset_y, tf.float16)
print(dataset_x.shape)
print(dataset_y.shape)
BATCH_SIZE = 4
# def preprocess(x,y):
#     x = tf.cast(x, tf.float16)
#     x = tf.expand_dims(x, axis=-1)
#     y = tf.cast(y, tf.float16)
#     y = tf.expand_dims(y, axis=-1)
#     return x,y


# dataset = tf.data.Dataset.from_tensor_slices((dataset_x, dataset_y))
# dataset = dataset.map(preprocess, tf.data.AUTOTUNE)
# dataset = dataset.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
pltcallback = PlotCallback(volume_N)
history = model.fit(dataset_x, dataset_y, batch_size=16, epochs=200, validation_split=0.2, shuffle=True,
                    callbacks=[pltcallback])

ckpt = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, "result/ckpt/", max_to_keep=10)
ckpt_manager.save()

# Convert to DataFrame
hist_df = pd.DataFrame(history.history)
# Save to CSV
hist_df.to_csv("result/training_history.csv", index=True)