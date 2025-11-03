import cv2
import torch
from torch import nn
from torchvision import transforms, models
import numpy as np
from collections import deque

PATH = r'C:\Users\SMILE\Downloads\Multi-Task-Learning-for-Images-age-gender-main\Multi-Task-Learning-for-Images-age-gender-main\state_dict_model.pt'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
IMAGE_SIZE = 224

# Store previous predictions to stabilize output
age_history = deque(maxlen=15)
gender_history = deque(maxlen=15)

def preprocess_image(im):
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    im = torch.tensor(im).permute(2,0,1).float()
    im = transform(im / 255.)
    return im.unsqueeze(0)

def get_model():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )

    class AgeGenderClassifier(nn.Module):
        def __init__(self):
            super(AgeGenderClassifier, self).__init__()
            self.intermediate = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            self.age_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            self.gender_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            gender = self.gender_classifier(x)
            return gender, age

    model.classifier = AgeGenderClassifier()
    return model

model = get_model()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

while True:
    ret, img = cap.read()
    if not ret:
        print("⚠️ Camera not accessible")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face_tensor = preprocess_image(face)

        with torch.no_grad():
            gender_pred, age_pred = model(face_tensor)

        gender_val = gender_pred.item()
        age_val = age_pred.item()

        age_val = int(age_val * 80)
        gender_label = "Female" if gender_val > 0.5 else "Male"

        # Add to history
        age_history.append(age_val)
        gender_history.append(gender_label)

        # Stable outputs
        smoothed_age = int(np.mean(age_history))
        stable_gender = max(set(gender_history), key=gender_history.count)

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f"{stable_gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Age: {smoothed_age}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Age & Gender Prediction (Stabilized)', img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
