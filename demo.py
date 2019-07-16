import os
import torch.utils.data.dataset
from PIL import Image
from torchvision import transforms
from network import Reconstructor
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model = Reconstructor()
    model.load_state_dict(torch.load('./190713_style_encode.pth.tar', map_location='cpu')['model_state_dict'])
    im1 = Image.open('./0d03083342661.56276fe897e50.png')
    im2 = Image.open('./0cbe7d22519513.56314036779c9.jpg')

    im1 = im1.convert('RGB')
    im2 = im2.convert('RGB')

    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    im1 = data_transforms(im1)
    im2 = data_transforms(im2)

    result = model.reconstruct(im1.unsqueeze(0), im2.unsqueeze(0))

    for i, item in enumerate(result):
        plt.subplot(3, 2, i + 1)
        plt.imshow(transforms.ToPILImage()(item.squeeze(0)))
        plt.axis('off')

    plt.show()
