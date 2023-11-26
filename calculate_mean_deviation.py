import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


ROOT = './data/images/'

transform = transforms.Compose([
    transforms.ToTensor(),
])

custom_dataset = ImageFolder(root=ROOT, transform=transform)

mean = torch.zeros(3)
std = torch.zeros(3)

for img, _ in tqdm(custom_dataset):
    mean += img.mean(dim=(1, 2))
    std += img.std(dim=(1, 2))

mean /= len(custom_dataset)
std /= len(custom_dataset)

print(f"Mean: {mean}, Standard Deviation: {std}")
