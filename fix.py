from tensorflow.keras.models import load_model

model = load_model("vgg16_custom_model.keras", compile=False)

model.save("fixed_model.keras")