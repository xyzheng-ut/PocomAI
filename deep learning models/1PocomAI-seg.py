import os

# Because of the use of tf.image.ssim in the loss,
# this example requires TensorFlow. The rest of the code
# is backend-agnostic.
os.environ["KERAS_BACKEND"] = "tensorflow"

from glob import glob
import matplotlib.pyplot as plt

import keras_hub
import tensorflow as tf
import keras
from keras import layers, ops

from pathlib import Path

import numpy as np, cv2, random
from scipy.ndimage import gaussian_filter, rotate
import pandas as pd



keras.config.disable_traceback_filtering()

IMAGE_SIZE = 128
BATCH_SIZE = 8
OUT_CLASSES = 1
TRAIN_SPLIT_RATIO = 0.90


print(tf.__version__)
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.list_logical_devices('GPU')

# ===================== Utilities =====================
def perlin_like(shape, scale=32):
    noise = np.random.randn(*shape).astype(np.float32)
    low = gaussian_filter(noise, sigma=scale).astype(np.float32)
    low -= low.min(); low /= (low.max() + 1e-6)
    return low

def motion_blur(img, ksize=15, angle=0.0):
    k = np.zeros((ksize, ksize), np.float32); k[ksize//2, :] = 1.0
    k = rotate(k, angle, reshape=False, order=1); k /= k.sum()
    return cv2.filter2D(img, -1, k)

def add_ring_artifacts(img, severity=0.02, n_rings=3, rng=None):
    if rng is None: rng = np.random.default_rng()
    h, w = img.shape; cy, cx = h/2.0, w/2.0
    Y, X = np.ogrid[:h, :w]; r = np.sqrt((Y-cy)**2 + (X-cx)**2)
    field = np.zeros_like(img, np.float32)
    for _ in range(n_rings):
        r0 = rng.uniform(0.2*min(h,w)/2, 0.9*min(h,w)/2)
        width = rng.uniform(3, 12)
        ring = np.exp(-0.5*((r-r0)/width)**2)
        amp = rng.uniform(-severity, severity)
        field += amp * ring
    return np.clip(img + field, 0, None)

def gather_npzs(path_like) -> list[Path]:
    p = Path(path_like)
    if p.is_dir():
        return sorted(p.glob("*.npz"))
    if p.is_file() and p.suffix.lower() == ".npz":
        return [p]
    raise FileNotFoundError("Provide a .npz file or a directory containing .npz files.")

# ===================== CT synthesis (slice by index) =====================
def synthesize_ct_slice_by_index_np(vol_path: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a cubic binary volume from .npz {'vol': (D,D,D)}, synthesize slice at index `idx`.
    Returns (ct_slice, mask_slice) as float32 in [0,1], shape (H,W,1).
    """
    with np.load(vol_path, allow_pickle=False) as z:
        vol01 = z["vol"]
    assert vol01.ndim == 3 and vol01.shape[0] == vol01.shape[1] == vol01.shape[2], "Expected cubic volume"
    D = vol01.shape[0]
    vol01 = vol01.astype(np.bool_) / 10.
    if not (0 <= idx < D):
        raise IndexError(f"idx {idx} out of bounds for D={D}")

    # m = (vol01[idx] > 0).astype(np.float32)  # 0/1 mask
    m = vol01[idx]
    H, W = m.shape
    mu_matrix, mu_filler = 0.45, 0.85  # arbitrary attenuation units

    mu = mu_matrix * (1 - m) + mu_filler * m  # base contrast
    lowfreq = perlin_like((H, W), scale=48) * 0.10  # bias field
    hi_bg = gaussian_filter(np.random.randn(H, W).astype(np.float32), 0.6) * 0.03
    hi_fil = gaussian_filter(np.random.randn(H, W).astype(np.float32), 0.9) * 0.02
    texture = (1 - m) * hi_bg + m * hi_fil

    img = mu + texture + (lowfreq - 0.05)  # combine
    img = gaussian_filter(img, sigma=0.8)  # PSF blur
    img = 0.85 * img + 0.15 * motion_blur(img, 21, 15)  # mild streaks

    # Poisson-like measurement and log-transform
    I0 = 4000.0
    photons = np.random.poisson(I0 * np.exp(-img)).astype(np.float32)
    mu_hat = -np.log(np.clip(photons / I0, 1e-6, 1.0))
    img = 0.6 * img + 0.4 * mu_hat

    # ring artifacts
    img = add_ring_artifacts(img, severity=0.03, n_rings=4)

    # normalize, slight gamma, tiny salt/pepper defects
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = np.power(img, 0.95)
    rng = np.random.default_rng()
    num_sp = int(0.0005 * H * W)
    coords = (rng.integers(0, H, num_sp), rng.integers(0, W, num_sp))
    img_sp = img.copy()
    img_sp[coords] = rng.choice([0.0, 1.0], size=num_sp)
    img = np.clip(img * 0.996 + img_sp * 0.004, 0, 1)

    # expand channels
    img = img.astype(np.float32)[..., None]
    # m = keras.utils.to_categorical((m*10).astype(np.int32), 2)
    m = m.astype(np.float32)[..., None]
    m = tf.cast(m*10, tf.float32)
    return img, m  # tf.tile(img, [1, 1, 3]), m


def _py_synthesize_indexed(path_tensor: tf.Tensor, idx_tensor: tf.Tensor):
    # Both tensors are int32/string; keep everything int32
    path = path_tensor.numpy().decode("utf-8")
    idx  = int(idx_tensor.numpy())  # handles int32 just fine
    img, m = synthesize_ct_slice_by_index_np(path, idx)
    return img, m

# ===================== Dataset builders (INT32 ONLY) =====================
def scan_files_with_ranges(files: list[Path], margin: int = 20):
    """
    Return a list of (path_str, start_int32, end_int32) with indices [start, end).
    Skips volumes that are too small for the margin.
    """
    triples: list[tuple[str, int, int]] = []
    for p in files:
        with np.load(str(p), allow_pickle=False) as z:
            vol = z["vol"]
        D = int(vol.shape[0])
        start = int(margin)
        end = int(max(margin, D - margin))
        if end - start <= 0:
            continue
        triples.append((str(p), start, end))
    if not triples:
        raise RuntimeError("No valid volumes with interior slices found.")
    return triples

def file_to_slice_pairs_ds(path: tf.Tensor, start: tf.Tensor, end: tf.Tensor) -> tf.data.Dataset:
    """
    Build a dataset of (path, idx) for idx in [start, end), all int32.
    Avoids Dataset.repeat(count) to sidestep int64 coercion.
    """
    start32 = tf.cast(start, tf.int32)
    end32   = tf.cast(end,   tf.int32)

    # indices [start, end) as int32 tensor -> dataset
    idxs = tf.data.Dataset.from_tensor_slices(tf.range(start32, end32, dtype=tf.int32))

    # Pair each idx with the same path
    return idxs.map(lambda idx: (path, idx), num_parallel_calls=tf.data.AUTOTUNE)


def make_dataset(files: list[Path], batch_size=8, shuffle=True, seed= 42) -> tf.data.Dataset:
    triples = scan_files_with_ranges(files, margin=20)
    paths, starts, ends = zip(*triples)

    files_ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(list(paths),  dtype=tf.string),
        tf.constant(list(starts), dtype=tf.int32),
        tf.constant(list(ends),   dtype=tf.int32),
    ))

    ds = files_ds.interleave(
        lambda p, s, e: file_to_slice_pairs_ds(p, s, e),
        cycle_length=min(8, len(triples)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=4096, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, i: tf.py_function(_py_synthesize_indexed, [p, i], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(lambda x, y: (tf.ensure_shape(x, [None, None, 1]), tf.ensure_shape(y, [None, None, 1])))
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds

# ===================== Split + main =====================
def train_val_split(files: list[Path], val_ratio=0.2, seed=123):
    files = list(files)
    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    val_files = files[:n_val]
    train_files = files[n_val:]
    return train_files, val_files



def display(display_list):
    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]), cmap="gray")
        plt.axis("off")
    plt.show()


npz_root = "/home/rc/pythonProject_img23d/data/vox_npy256"
os.makedirs("result", exist_ok=True)
os.makedirs("result/pred", exist_ok=True)
all_npz = gather_npzs(npz_root)
print(len(all_npz))
if not all_npz:
    raise RuntimeError(f"No .npz files found under: {npz_root}")

train_files, val_files = train_val_split(all_npz, val_ratio=0.2, seed=123)

train_dataset = make_dataset(train_files, batch_size=BATCH_SIZE, shuffle=True, seed=42)
val_dataset   = make_dataset(val_files,   batch_size=BATCH_SIZE, shuffle=True)
print(train_dataset)
print(f"#files -> train: {len(train_files)} | val: {len(val_files)}")

for x, y in train_dataset.take(1):
    print("train batch shapes:", x.shape, y.shape)
    print(np.max(x))
    print(np.max(y))


def basic_block(x_input, filters, stride=1, down_sample=None, activation=None):
    """Creates a residual(identity) block with two 3*3 convolutions."""
    residual = x_input

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(
        x_input
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding="same", use_bias=False)(
        x
    )
    x = layers.BatchNormalization()(x)

    if down_sample is not None:
        residual = down_sample

    x = layers.Add()([x, residual])

    if activation is not None:
        x = layers.Activation(activation)(x)

    return x


def convolution_block(x_input, filters, dilation=1):
    """Apply convolution + batch normalization + relu layer."""
    x = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=dilation)(x_input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def segmentation_head(x_input, out_classes, final_size):
    """Map each decoder stage output to model output classes."""
    x = layers.Conv2D(out_classes, kernel_size=(3, 3), padding="same")(x_input)

    if final_size is not None:
        x = layers.Resizing(final_size[0], final_size[1])(x)

    return x


def get_resnet_block(resnet, block_num):
    """Extract and return a ResNet-34 block."""
    extractor_levels = ["P2", "P3", "P4", "P5"]
    num_blocks = resnet.stackwise_num_blocks
    if block_num == 0:
        x = resnet.get_layer("pool1_pool").output
    else:
        x = resnet.pyramid_outputs[extractor_levels[block_num - 1]]
    y = resnet.get_layer(f"stack{block_num}_block{num_blocks[block_num]-1}_add").output
    return keras.models.Model(
        inputs=x,
        outputs=y,
        name=f"resnet_block{block_num + 1}",
    )


def basnet_predict(input_shape, out_classes):
    """BASNet Prediction Module, it outputs coarse label map."""
    filters = 64
    num_stages = 2

    x_input = layers.Input(input_shape)

    # -------------Encoder--------------
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    resnet = keras_hub.models.ResNetBackbone(
        input_conv_filters=[64],
        input_conv_kernel_sizes=[7],
        stackwise_num_filters=[64, 80, 96, 128],
        stackwise_num_blocks=[3, 4, 6, 3],
        stackwise_num_strides=[1, 2, 2, 2],
        block_type="basic_block",
    )

    encoder_blocks = []
    for i in range(num_stages):
        if i < 4:  # First four stages are adopted from ResNet-34 blocks.
            x = get_resnet_block(resnet, i)(x)
            encoder_blocks.append(x)
            x = layers.Activation("relu")(x)
        else:  # Last 2 stages consist of three basic resnet blocks.
            x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
            x = basic_block(x, filters=filters * 8, activation="relu")
            x = basic_block(x, filters=filters * 8, activation="relu")
            x = basic_block(x, filters=filters * 8, activation="relu")
            encoder_blocks.append(x)

    # -------------Bridge-------------
    x = convolution_block(x, filters=filters * 8, dilation=2)
    x = convolution_block(x, filters=filters * 8, dilation=2)
    x = convolution_block(x, filters=filters * 8, dilation=2)
    encoder_blocks.append(x)

    # -------------Decoder-------------
    decoder_blocks = []
    for i in reversed(range(num_stages)):
        if i != (num_stages - 1):  # Except first, scale other decoder stages.
            shape = x.shape
            x = layers.Resizing(shape[1] * 2, shape[2] * 2)(x)

        x = layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        decoder_blocks.append(x)

    decoder_blocks.reverse()  # Change order from last to first decoder stage.
    decoder_blocks.append(encoder_blocks[-1])  # Copy bridge to decoder.

    # -------------Side Outputs--------------
    decoder_blocks = [
        segmentation_head(decoder_block, out_classes, input_shape[:2])
        for decoder_block in decoder_blocks
    ]

    return keras.models.Model(inputs=x_input, outputs=decoder_blocks)


def basnet_rrm(base_model, out_classes):
    """BASNet Residual Refinement Module(RRM) module, output fine label map."""
    num_stages = 2
    filters = 64

    x_input = base_model.output[0]

    # -------------Encoder--------------
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    encoder_blocks = []
    for _ in range(num_stages):
        x = convolution_block(x, filters=filters)
        encoder_blocks.append(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # -------------Bridge--------------
    x = convolution_block(x, filters=filters)

    # -------------Decoder--------------
    for i in reversed(range(num_stages)):
        shape = x.shape
        x = layers.Resizing(shape[1] * 2, shape[2] * 2)(x)
        x = layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters)

    x = segmentation_head(x, out_classes, None)  # Segmentation head.

    # ------------- refined = coarse + residual
    x = layers.Add()([x_input, x])  # Add prediction + refinement output

    return keras.models.Model(inputs=[base_model.input], outputs=[x])


class BASNet(keras.Model):
    def __init__(self, input_shape, out_classes):
        """BASNet, it's a combination of two modules
        Prediction Module and Residual Refinement Module(RRM)."""

        # Prediction model.
        predict_model = basnet_predict(input_shape, out_classes)
        # Refinement model.
        refine_model = basnet_rrm(predict_model, out_classes)

        output = refine_model.outputs  # Combine outputs.
        output.extend(predict_model.output)

        # Activations.
        output = [layers.Activation("sigmoid")(x) for x in output]
        super().__init__(inputs=predict_model.input, outputs=output)

        self.smooth = 1.0e-9
        # Binary Cross Entropy loss.
        self.cross_entropy_loss = keras.losses.BinaryCrossentropy()
        # Structural Similarity Index value.
        self.ssim_value = tf.image.ssim
        # Jaccard / IoU loss.
        self.iou_value = self.calculate_iou

    def calculate_iou(
        self,
        y_true,
        y_pred,
    ):
        """Calculate intersection over union (IoU) between images."""
        intersection = ops.sum(ops.abs(y_true * y_pred), axis=[1, 2, 3])
        union = ops.sum(y_true, [1, 2, 3]) + ops.sum(y_pred, [1, 2, 3])
        union = union - intersection
        return ops.mean((intersection + self.smooth) / (union + self.smooth), axis=0)

    def compute_loss(self, x, y_true, y_pred, sample_weight=None, training=False):
        total = 0.0
        for y_pred_i in y_pred:  # y_pred = refine_model.outputs + predict_model.output
            cross_entropy_loss = self.cross_entropy_loss(y_true, y_pred_i)

            ssim_value = self.ssim_value(y_true, y_pred_i, max_val=1.0)
            ssim_loss = ops.mean(1 - ssim_value + self.smooth, axis=0)

            iou_value = self.iou_value(y_true, y_pred_i)
            iou_loss = 1 - iou_value

            # Add all three losses.
            total += cross_entropy_loss + ssim_loss + iou_loss
        return total


basnet_model = BASNet(
    input_shape=[IMAGE_SIZE, IMAGE_SIZE, 1], out_classes=OUT_CLASSES
)  # Create model.
basnet_model.summary()  # Show model summary.

optimizer = keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
# Compile model.
basnet_model.compile(
    optimizer=optimizer,
    metrics=[keras.metrics.MeanAbsoluteError(name="mae") for _ in basnet_model.outputs],
)


def normalize_output(prediction):
    max_value = np.max(prediction)
    min_value = np.min(prediction)
    return (prediction-min_value)/(max_value - min_value)


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions, axis=0)
    predictions = np.argmax(predictions, axis=-1)
    print(np.max(predictions))
    print(np.min(predictions))
    predictions = (predictions*255).astype(np.uint8)
    return predictions


class PlotCallback(keras.callbacks.Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def on_epoch_end(self,epoch, log={}):
        for x, y in self.data.take(2):
            pred_mask = self.model.predict(x)
            pred_mask = normalize_output(pred_mask)
            _, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 7.2))
            for i in range(5):
                img_xct = x[i]
                img_gt = y[i]
                img_pred = pred_mask[0][i]

                axes[0, i].imshow(img_xct[:, :, 0], cmap="gray")
                axes[1, i].imshow(img_gt, cmap="gray")
                axes[2, i].imshow(img_pred, cmap="gray")

            for ax in axes.ravel():
                ax.set_axis_off()
            plt.savefig(f"result/pred/Val_pred_{epoch}.png", dpi=600)
            plt.close()
#
#
pltcallback = PlotCallback(val_dataset)
history= basnet_model.fit(train_dataset, validation_data=val_dataset, epochs=100,
                          callbacks=[pltcallback])
ckpt = tf.train.Checkpoint(model=basnet_model)
ckpt_manager = tf.train.CheckpointManager(ckpt, "result/ckpt/", max_to_keep=2)
ckpt_manager.save()

# Convert to DataFrame
hist_df = pd.DataFrame(history.history)
# Save to CSV
hist_df.to_csv("result/training_history.csv", index=True)
