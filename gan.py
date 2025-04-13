import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import os

np.random.seed(42)
tf.random.set_seed(42)

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
        layers.Conv2D(1, (7, 7), padding="same", activation="linear")
    ])
    return model

generator = build_generator()

def build_discriminator():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

discriminator = build_discriminator()

discriminator_optimizer = optimizers.Adam(learning_rate=0.00005, beta_1=0.5)
generator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

gan_input = keras.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = keras.Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer=generator_optimizer)

(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

epochs = 10000
batch_size = 128

if os.path.exists("generated_images"):
    for filename in os.listdir("generated_images"):
        file_path = os.path.join("generated_images", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    os.makedirs("generated_images")

if not os.path.exists("generated_images"):
    os.makedirs("generated_images")

for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]

    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict(noise)
    print(f"[DEBUG] Epoch {epoch} - Fake image range: min={np.min(fake_images)}, max={np.max(fake_images)}")

    img = np.clip(fake_images[0, :, :, 0], 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save("generated_images/test_output.png")

    real_labels = np.random.uniform(0.9, 1.0, size=(batch_size, 1))
    fake_labels = np.random.uniform(0.0, 0.1, size=(batch_size, 1))

    with tf.GradientTape() as disc_tape:
        real_loss = tf.keras.losses.binary_crossentropy(real_labels, discriminator(real_images))
        fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, discriminator(fake_images))
        disc_loss = (real_loss + fake_loss) / 2

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    noise = np.random.normal(0, 1, (batch_size, 100))
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise)
        gen_loss = tf.keras.losses.binary_crossentropy(np.ones((batch_size, 1)), discriminator(generated_images))

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: D Loss: {disc_loss.numpy()}, G Loss: {gen_loss.numpy()}")

        generated_images = generator.predict(noise)
        plt.figure(figsize=(10, 2))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            image = np.clip(generated_images[i, :, :, 0], 0.0, 1.0)
            image = (image * 255).astype(np.uint8)
            print(f"Generated image range: min={np.min(generated_images[i])}, max={np.max(generated_images[i])}")
            plt.imshow(image, cmap="gray")
            plt.axis("off")

        filename = f"generated_images/epoch_{epoch}.png"
        plt.savefig(filename)
        plt.close()

# Save model weights after training
generator.save_weights("generator.weights.h5")
discriminator.save_weights("discriminator.weights.h5")
gan.save_weights("gan.weights.h5")
