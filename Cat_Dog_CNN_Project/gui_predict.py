import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("cat_dog_model.h5")

IMG_SIZE = 64

def predict_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction < 0.5:
        result = "CAT"
        accuracy = (1 - prediction) * 100
    else:
        result = "DOG"
        accuracy = prediction * 100

    result_label.config(
        text=f"Prediction: {result}\nAccuracy: {accuracy:.2f}%"
    )

    img_display = ImageTk.PhotoImage(img.resize((200, 200)))
    image_label.config(image=img_display)
    image_label.image = img_display


# ---------- GUI ----------
root = tk.Tk()
root.title("Cat vs Dog CNN Classifier")
root.geometry("400x500")
root.resizable(False, False)

btn = tk.Button(root, text="Select Image", command=predict_image, font=("Arial", 12))
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
