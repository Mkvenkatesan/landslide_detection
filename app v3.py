import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import matplotlib.pyplot as plt

class ImagePredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Landslide Occurence Predictior")

        self.model_path = tk.StringVar(value="path/to/model")
        self.class_path = tk.StringVar(value="path/to/classnames")
        self.image_path = tk.StringVar(value="path/to/image")

        self.model_label = tk.Label(master, text="Model Path:")
        self.model_label.grid(row=0, column=0)
        self.model_entry = tk.Entry(master, textvariable=self.model_path, width=50)
        self.model_entry.grid(row=0, column=1)
        self.model_button = tk.Button(master, text="Browse", command=self.browse_model)
        self.model_button.grid(row=0, column=2)

        self.class_label = tk.Label(master, text="Class Path:")
        self.class_label.grid(row=1, column=0)
        self.class_entry = tk.Entry(master, textvariable=self.class_path, width=50)
        self.class_entry.grid(row=1, column=1)
        self.class_button = tk.Button(master, text="Browse", command=self.browse_class)
        self.class_button.grid(row=1, column=2)

        self.image_label = tk.Label(master, text="Image Path:")
        self.image_label.grid(row=2, column=0)
        self.image_entry = tk.Entry(master, textvariable=self.image_path, width=50)
        self.image_entry.grid(row=2, column=1)
        self.image_button = tk.Button(master, text="Browse", command=self.browse_image)
        self.image_button.grid(row=2, column=2)

        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.grid(row=3, column=1)

    def browse_model(self):
        file_path = filedialog.askopenfilename(title="Select Model File")
        self.model_path.set(file_path)

    def browse_class(self):
        file_path = filedialog.askopenfilename(title="Select Class File")
        self.class_path.set(file_path)

    def browse_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File")
        self.image_path.set(file_path)

    def measure_inference_time(self, model_path, image_path):
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        input_shape = model.input_shape[1:]

        img = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255

        start_time = time.time()
        result = model.predict(img)
        end_time = time.time()

        inference_time = end_time - start_time
        return inference_time

    def predict(self):
        model_path = self.model_path.get()
        class_path = self.class_path.get()
        image_path = self.image_path.get()

        try:
            inference_time = self.measure_inference_time(model_path, image_path)

            model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
            input_shape = model.input_shape[1:]

            img = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255

            result = model.predict(img)
            lines = [line.replace("\n", "") for line in open(class_path, "r").readlines()]
            classes = {i: lines[i] for i in range(len(lines))}
            prediction_probability = {i: j for i, j in zip(classes.values(), result.tolist()[0])}

            messagebox.showinfo("Landslide Occurence Prediction Rate", str(prediction_probability))

            # Plot performance analysis graph
            plt.bar(["Inference Time"], [inference_time], align='center', alpha=0.5)
            plt.ylabel('Time (s)')
            plt.title('Inference Time')
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


root = tk.Tk()
app = ImagePredictorApp(root)
root.mainloop()
