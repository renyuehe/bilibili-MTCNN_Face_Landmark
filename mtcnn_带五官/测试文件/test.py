from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader


# data_celebA = datasets.CelebA(r"D:\Desktop\data\CelebA", transform=transforms.ToTensor(), download=False)
# dataloader_celebA = DataLoader(dataset=data_celebA,batch_size=10,shuffle=True)

Celeba_dataset = datasets.ImageFolder(r"D:\Desktop\data\CelebA\Img\img_align_celeba_png.7z\img_align_celeba_png", transform=transforms.ToTensor())

# for i, (img, tag) in enumerate(dataloader_celebA):
#     print("-----------------")
#     print(img.shape)
#     print(tag.shape)