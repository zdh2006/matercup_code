from torchvision import transforms
from PIL import Image

img_path="练手数据集/train/ants_image/0013035.jpg"
Img=Image.open(img_path)

tensor_trans=transforms.ToTensor()
tensor_img = tensor_trans(Img)

print(tensor_img)