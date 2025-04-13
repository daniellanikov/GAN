import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Input, Model
import tf2onnx

# Rebuild the generator architecture
def build_generator():
    model = keras.Sequential([
        keras.Input(shape=(100,)),
        layers.Dense(128 * 7 * 7, activation="relu"),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="tanh")
    ])
    return model

# Instantiate and load weights
generator = build_generator()
generator.load_weights("generator.weights.h5")

# Wrap as functional model for tf2onnx compatibility
inp = Input(shape=(100,), name="input")
out = generator(inp)
model = Model(inputs=inp, outputs=out)

# Convert to ONNX
spec = (tf.TensorSpec((None, 100), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path="generator.onnx")

print("✅ Generator model exported to generator.onnx")
