import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


TRAIN = True

def main():
    model = models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)

    num_classes = 4
    in_features = model.classifier[1].in_features  # Get the input features for the last layer
    model.classifier[1] = nn.Linear(in_features, num_classes)  # Replace the last layer

    model.load_state_dict(torch.load('mobileNet_fine_tuned.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    transform_train = transforms.Compose(
        [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.GaussianBlur((5, 5), sigma=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ElasticTransform(),
            transforms.ToTensor(),
            transforms.Normalize([0.4093, 0.3851, 0.3808], [0.2469, 0.2507, 0.2426]),])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize([0.4093, 0.3851, 0.3808], [0.2469, 0.2507, 0.2426]),])

    trainset = torchvision.datasets.ImageFolder(root='./data/images/', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

    testset = torchvision.datasets.ImageFolder(root='./data/test_images2/', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True)

    while True:
        if TRAIN:
            model.train()
            for idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = F.softmax(model(inputs), dim=1)
                loss = criterion(outputs, labels)
                print(loss.item(), outputs, labels)
                print("Accuracy: ", (outputs.argmax(dim=1) == labels).float().mean())
                loss.backward()
                optimizer.step()

                if idx % 10 == 9:
                    print("Saving model")
                    torch.save(model.state_dict(), 'mobileNet_fine_tuned.pth')

        if not TRAIN:
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    outputs = F.softmax(outputs)
                    loss = criterion(outputs, labels)
                    print(loss.item(), outputs, outputs.argmax(dim=1), labels)
                    print("Accuracy: ", (outputs.argmax(dim=1) == labels).float().mean())


if __name__ == '__main__':
    main()
