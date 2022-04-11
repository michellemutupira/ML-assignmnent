import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np

model = VGG16(weights='imagenet', include_top='True')

print("Done...")


def identify(): 

  img = image.load_img("car.jpg", color_mode='rgb', target_size=(224,224)) 

  # Resizing to fit into VGGNET
  x = image.img_to_array(img)
  x.shape
  x = np.expand_dims(x, axis=0)

  # Using Pre-Trained model for prediction
  x = preprocess_input(x)
  features = model.predict(x)
  p = decode_predictions(features)
  return p

label = identify()[0][0][1]
print(f"A {label} was identified in the image...")

identify()
