from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score

# Load model 
model = load_model("models/rice_model.h5")
IMG_SIZE = (250, 250)
test_dir = "data/test"

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(test_dir, target_size=IMG_SIZE,batch_size=32, class_mode='categorical', shuffle=False)

# Predict
preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

# Evaluation
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")


