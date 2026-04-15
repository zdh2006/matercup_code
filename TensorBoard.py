from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


writer=SummaryWriter('logs')
image_path="练手数据集/train/ants_image/0013035.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
writer.add_image("test",img_array,dataformats='HWC')
# writer.add_scalar()
# for i in range(100):
#     writer.add_scalar('y=x',i,i)



# writer.close()
