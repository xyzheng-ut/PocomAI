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
import trimesh


keras.config.disable_traceback_filtering()

IMAGE_SIZE = 128
BATCH_SIZE = 8
OUT_CLASSES = 101
TRAIN_SPLIT_RATIO = 0.90

print(tf.__version__)
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.list_logical_devices('GPU')


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

    npz_root_files = sorted(glob(npz_root+"/*.npz"))
    npz_root_files = npz_root_files[start:stop:step]
    volume_Y = np.zeros((len(npz_root_files), VOXEL_SIZE,VOXEL_SIZE,VOXEL_SIZE))
    volume_X = np.zeros((len(npz_root_files), VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE))
    for i in range(len(npz_root_files)):
        with np.load(npz_root_files[i], allow_pickle=False) as z:
            volume = z["vol"]
            # volume[volume>1] = 1
        volume_Y[i,:,:,:] = volume
    volume_X[volume_Y>1] = 1
    return tf.expand_dims(volume_X, axis=-1), tf.expand_dims(volume_Y, axis=-1)


def basic_block(x_input, filters, stride=1, down_sample=None, activation=None):
    """Creates a residual(identity) block with two 3*3 convolutions."""
    residual = x_input

    x = layers.Conv3D(filters, (3, 3, 3), strides=stride, padding="same", use_bias=False)(
        x_input
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv3D(filters, (3, 3, 3), strides=(1, 1, 1), padding="same", use_bias=False)(
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
    x = layers.Conv3D(filters, (3, 3, 3), padding="same", dilation_rate=dilation)(x_input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def segmentation_head(x_input, out_classes, final_size):
    """Map each decoder stage output to model output classes."""
    x = layers.Conv3D(out_classes, kernel_size=(3, 3, 3), padding="same")(x_input)

    if final_size is not None:
        x = layers.Resizing(final_size[0], final_size[1])(x)

    return x


def basic_block_3d(x, filters, stride=1, name=None):
    """3D Basic ResNet block (for ResNet-34)."""
    shortcut = x

    # First conv block
    x = layers.Conv3D(filters, 3, strides=stride, padding='same',
                      name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)

    # Second conv block
    x = layers.Conv3D(filters, 3, strides=1, padding='same',
                      name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)

    # Shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, 1, strides=stride, padding='same',
                                 name=f'{name}_shortcut_conv')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    x = layers.Add(name=f'{name}_add')([shortcut, x])
    x = layers.Activation('relu', name=f'{name}_relu2')(x)

    return x


def create_resnet34_3d(input_shape,
                       stackwise_num_filters=[64, 128, 256, 512],
                       stackwise_num_blocks=[3, 4, 6, 3],
                       stackwise_num_strides=[1, 2, 2, 2]):
    """Create a 3D ResNet-34 backbone."""

    inputs = layers.Input(shape=input_shape)

    # Initial conv layer
    x = layers.Conv3D(64, 7, strides=2, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling3D(pool_size=3, strides=2, padding='same',
                            name='pool1_pool')(x)

    # ResNet blocks
    pyramid_outputs = {}
    for stack_idx, (num_filters, num_blocks, stride) in enumerate(
            zip(stackwise_num_filters, stackwise_num_blocks, stackwise_num_strides)
    ):
        for block_idx in range(num_blocks):
            block_stride = stride if block_idx == 0 else 1
            x = basic_block_3d(
                x,
                num_filters,
                stride=block_stride,
                name=f'stack{stack_idx}_block{block_idx}'
            )

        # Store pyramid outputs
        pyramid_outputs[f'P{stack_idx + 2}'] = x

    model = keras.models.Model(inputs=inputs, outputs=x)
    model.pyramid_outputs = pyramid_outputs
    model.stackwise_num_blocks = stackwise_num_blocks

    return model


def get_resnet_block_3d(resnet, block_num):
    """Extract and return a 3D ResNet block."""
    extractor_levels = ["P2", "P3", "P4", "P5"]
    num_blocks = resnet.stackwise_num_blocks

    if block_num == 0:
        x_input = resnet.get_layer("pool1_pool").output
    else:
        x_input = resnet.pyramid_outputs[extractor_levels[block_num - 1]]

    y_output = resnet.get_layer(
        f"stack{block_num}_block{num_blocks[block_num] - 1}_add"
    ).output

    return keras.models.Model(
        inputs=x_input,
        outputs=y_output,
        name=f"resnet_block{block_num + 1}",
    )


def basnet_predict(input_shape, out_classes):
    """BASNet Prediction Module, it outputs coarse label map."""
    filters = 64
    num_stages = 2

    x_input = layers.Input(input_shape)

    # -------------Encoder--------------
    x = layers.Conv3D(filters, kernel_size=(3, 3, 3), padding="same")(x_input)
    resnet_3d = create_resnet34_3d(
        input_shape=(512, 512, 512, 1),  # (depth, height, width, channels)
        stackwise_num_filters=[32, 80, 96, 128],
        stackwise_num_blocks=[3, 4, 6, 3],
        stackwise_num_strides=[1, 2, 2, 2]
    )

    encoder_blocks = []
    for i in range(num_stages):
        if i < 4:  # First four stages are adopted from ResNet-34 blocks.
            print("i:", i)
            print("x shape", x.shape)
            x = get_resnet_block_3d(resnet_3d, i)(x)
            # x = get_resnet_block(resnet, i)(x)
            encoder_blocks.append(x)
            x = layers.Activation("relu")(x)
        else:  # Last 2 stages consist of three basic resnet blocks.
            x = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)
            x = basic_block(x, filters=filters * 2, activation="relu")
            x = basic_block(x, filters=filters * 2, activation="relu")
            x = basic_block(x, filters=filters * 2, activation="relu")
            encoder_blocks.append(x)

    # -------------Bridge-------------
    x = convolution_block(x, filters=filters * 2, dilation=2)
    x = convolution_block(x, filters=filters * 2, dilation=2)
    x = convolution_block(x, filters=filters * 2, dilation=2)
    encoder_blocks.append(x)

    # -------------Decoder-------------
    decoder_blocks = []
    for i in reversed(range(num_stages)):
        if i != (num_stages - 1):  # Except first, scale other decoder stages.
            # x = layers.Resizing(shape[1] * 2, shape[2] * 2)(x)
            x = keras.layers.UpSampling3D(size=(2, 2, 2), data_format="channels_last")(x)

        x = layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters * 4)
        x = convolution_block(x, filters=filters * 4)
        x = convolution_block(x, filters=filters * 4)
        decoder_blocks.append(x)

    decoder_blocks.reverse()  # Change order from last to first decoder stage.
    decoder_blocks.append(encoder_blocks[-1])  # Copy bridge to decoder.

    # -------------Side Outputs--------------
    decoder_blocks = [
        segmentation_head(decoder_block, out_classes, input_shape[:3])
        for decoder_block in decoder_blocks
    ]

    return keras.models.Model(inputs=x_input, outputs=decoder_blocks)


def basnet_rrm(base_model, out_classes):
    """BASNet Residual Refinement Module(RRM) module, output fine label map."""
    num_stages = 2
    filters = 32

    x_input = base_model.output[0]

    # -------------Encoder--------------
    x = layers.Conv3D(filters, kernel_size=(3, 3, 3), padding="same")(x_input)

    encoder_blocks = []
    for _ in range(num_stages):
        x = convolution_block(x, filters=filters)
        encoder_blocks.append(x)
        x = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # -------------Bridge--------------
    x = convolution_block(x, filters=filters)

    # -------------Decoder--------------
    for i in reversed(range(num_stages)):
        # shape = x.shape
        # x = layers.Resizing(shape[1] * 2, shape[2] * 2)(x)
        x = keras.layers.UpSampling3D(size=(2, 2, 2), data_format="channels_last")(x)

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
        # self.cross_entropy_loss = keras.losses.categorical_crossentropy()
        # Structural Similarity Index value.
        self.ssim_value = tf.image.ssim
        # Jaccard / IoU loss.
        self.iou_value = self.calculate_iou

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            print(y_pred.shape)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y, y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        iou = self.calculate_iou(y, y_pred)
        # Return a dict mapping metric names to current value
        return {"iou": iou}

    def predict_step(self, x):
        # Adding dummy dimension using tf.expand_dims and converting to float32 using tf.cast
        # x = ops.cast(tf.expand_dims(x, axis=0), dtype="float32")
        # Passing low resolution image to model
        super_resolution_img = self(x, training=False)
        super_resolution_img = tf.math.argmax(super_resolution_img)
        # Clips the tensor from min(0) to max(255)
        # super_resolution_img = ops.clip(super_resolution_img, 0, 1)
        # Rounds the values of a tensor to the nearest integer
        # super_resolution_img = ops.round(super_resolution_img)
        # Removes dimensions of size 1 from the shape of a tensor and converting to uint8
        # super_resolution_img = ops.squeeze(
        #     ops.cast(super_resolution_img, dtype="uint8"), axis=0
        # )
        return super_resolution_img

    def calculate_iou(
        self,
        y_true,
        y_pred,
    ):
        """Calculate intersection over union (IoU) between images."""
        intersection = ops.sum(ops.abs(y_true * y_pred), axis=[1, 2, 3,4])
        union = ops.sum(y_true, [1, 2, 3,4]) + ops.sum(y_pred, [1, 2, 3,4])
        union = union - intersection
        return ops.mean((intersection + self.smooth) / (union + self.smooth), axis=0)

    def compute_loss(self, x, y_true, y_pred, sample_weight=None, training=False):
        total = 0.0
        for y_pred_i in y_pred:  # y_pred = refine_model.outputs + predict_model.output
            cross_entropy_loss = keras.losses.categorical_crossentropy(y_true, y_pred_i)

            ssim_value = self.ssim_value(y_true, y_pred_i, max_val=1.0)
            ssim_loss = ops.mean(1 - ssim_value + self.smooth, axis=0)

            iou_value = self.iou_value(y_true, y_pred_i)
            iou_loss = 1 - iou_value

            # Add all three losses.
            # print("cross_entropy_loss:", cross_entropy_loss)
            # print("ssim_loss:", ssim_loss)
            # print("iou_loss:", iou_loss)
            total += cross_entropy_loss + ssim_loss + iou_loss
        return total



# main
dataset_x, dataset_y = load_data("/home/rc/pythonProject_img23d/data/vox_npy128", VOXEL_SIZE=128, start=0, stop=150000, step=1)  # [N,64,64,64,1]
print("max y: ", np.max(dataset_y))
print("data x: ", dataset_x.shape)
print("data y: ", dataset_y.shape)
os.makedirs("result", exist_ok=True)
os.makedirs("result/pred", exist_ok=True)
dataset_x = tf.cast(dataset_x, tf.float16)
dataset_y = tf.cast(dataset_y, tf.float16)



basnet_model = BASNet(
    input_shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1], out_classes=OUT_CLASSES
)  # Create model.
basnet_model.summary()  # Show model summary.
xx = tf.random.uniform(shape=[2,128,128,128,1])
yy = basnet_model(xx)
print(yy.shape)
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
                # edges = cv2.Canny(img_pred, 100, 200)
                # overlay = cv2.cvtColor((np.array(img_xct[:, :, 0]) * 255).astype(np.uint8),
                #                        cv2.COLOR_GRAY2BGR)  # cv2.cvtColor((img_xct[:,:,0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # Convert to color for overlay
                # overlay[edges > 0] = [255, 0, 0]  # Draw edges in red
                # img_pred_bounday = overlay

                axes[0, i].imshow(img_xct[:, :, 0], cmap="gray")
                # axes[1, i].imshow(img_pred_bounday)
                axes[1, i].imshow(img_gt, cmap="gray")
                axes[2, i].imshow(img_pred, cmap="gray")

            for ax in axes.ravel():
                ax.set_axis_off()
            plt.savefig(f"result/pred/Val_pred_{epoch}.png", dpi=600)
            plt.close()
#
#
# pltcallback = PlotCallback(val_dataset)
history= basnet_model.fit(dataset_x, dataset_y, epochs=100,batch_size=8, validation_split=0.1)
                          # callbacks=[pltcallback])
ckpt = tf.train.Checkpoint(model=basnet_model)
ckpt_manager = tf.train.CheckpointManager(ckpt, "result/ckpt/", max_to_keep=10)
ckpt_manager.save()

# Convert to DataFrame
hist_df = pd.DataFrame(history.history)
# Save to CSV
hist_df.to_csv("result/training_history.csv", index=True)
#


