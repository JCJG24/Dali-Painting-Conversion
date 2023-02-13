import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the Salvador Dali painting dataset
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Preprocess the data by rescaling the pixel values
X_train = X_train / 255.

# Create the generator model
def generator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_shape=(100,)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(np.prod(X_train.shape[1:]), activation='tanh'),
        tf.keras.layers.Reshape(X_train.shape[1:])
    ])
    return model

# Create the discriminator model
def discriminator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=X_train.shape[1:]),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Compile the discriminator model
discriminator = discriminator_model()
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# Compile the generator model
generator = generator_model()

# Create the combined model
z = tf.keras.layers.Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)
combined = tf.keras.Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Train the GAN
def train(X_train, epochs=10000, batch_size=128):
    for epoch in range(epochs):
        # Select a random batch of images from the training set
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

    # Generate a batch of noise to use as input to the generator
    noise = np.random.normal(0, 1, (batch_size, 100))

    # Generate a batch of fake images using the generator
    fake_imgs = generator.predict(noise)

    # Train the discriminator on the real and fake images
    d_loss_real = discriminator.train_on_batch(imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator to fool the discriminator
    g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print the progress after every 100 epochs
    if epoch % 100 == 0:
        print("Epoch: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

# Train the GAN on the Salvador Dali painting dataset
train(X_train)

#Save the trained models
generator.save('dali_generator.h5')
discriminator.save('dali_discriminator.h5')

# Function to convert an input image to a Dali painting
def convert_to_dali(input_image):
    # Load the trained generator model
    generator = tf.keras.models.load_model('dali_generator.h5')
    
    # Preprocess the input image by rescaling the pixel values
    input_image = input_image / 255.
    
    # Generate a noise vector to use as input to the generator
    noise = np.random.normal(0, 1, (1, 100))
    
    # Use the generator to generate a Dali painting from the input image and noise vector
    dali_painting = generator.predict(noise)
    
    # Rescale the pixel values of the generated image to the range [0, 255]
    dali_painting = (dali_painting * 255).astype(np.uint8)
    
    return dali_painting

# Test the conversion function
input_image = X_train[0]
dali_painting = convert_to_dali(input_image)

# Plot the input image and the generated Dali painting
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(dali_painting[0], cmap='gray')
plt.axis('off')
plt.show()