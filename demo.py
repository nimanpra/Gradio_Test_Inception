import gradio as gr
import tensorflow as tf
import requests

inception_net = tf.keras.applications.InceptionV3() # load the model

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def classify_image(inp):
  inp = inp.reshape((-1, 299, 299, 3))
  inp = tf.keras.applications.inception_v3.preprocess_input(inp)
  prediction = inception_net.predict(inp).flatten()
  return {labels[i]: float(prediction[i]) for i in range(1000)}

image = gr.inputs.Image(shape=(299, 299))
label = gr.outputs.Label(num_top_classes=3)

examples = [['car1.jpg'], ['car2.jpg'], ['apple1.jpg'], ['apple2.jfif'], ['groom1.jpg'], ['groom2.jpg'], ['balloon1.jfif'], ['balloon2.jpg'], ['mask1.jpg'], ['mask2.jpg'], ['lion1.png'], ['lion2.png'], ['roomba1.jpg'], ['roomba2.jpg'], ['geese1.png'], ['geese2.png']]

gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples).launch()
