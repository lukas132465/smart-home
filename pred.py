import torchvision.models as models
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2


weights_path = 'mobileNet_fine_tuned.pth'
model = models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)

num_classes = 4
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

model.load_state_dict(torch.load(weights_path))

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize([0.4093, 0.3851, 0.3808], [0.2469, 0.2507, 0.2426])])
    frame_tensor = transform(frame)
    frame_tensor = frame_tensor.unsqueeze(0)

    output = F.softmax(model(frame_tensor))
    targets = ['Cross', 'Down', 'Other', 'Up']
    print(targets[output.argmax(dim=1)], output)
