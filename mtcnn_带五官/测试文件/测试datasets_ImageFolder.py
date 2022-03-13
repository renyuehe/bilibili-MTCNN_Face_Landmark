from torchvision import transforms
from torchvision import datasets

ret = datasets.ImageFolder(r"D:\Desktop\data\CelebA\Img\img_align_celeba_png.7z", transform=transforms.ToTensor())
print(ret)

for i , img in enumerate(ret):
    print(".........")
    print(i)
    print(img.shape)
