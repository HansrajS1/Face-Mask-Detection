import tensorflow as tf

model = tf.keras.models.load_model(
    "vgg16_custom_model.h5",
    compile=False
)

model.save("vgg16_custom_model2.keras")

print("Converted successfully!")