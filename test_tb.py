from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("logs")
image_path = "data/train/ants_image/6743948_2b8c096dda.jpg"
img_PIL = Image.open(image_path) #打开image_path图片
img_array = np.array(img_PIL)
writer.add_image("train",img_array,2,dataformats='HWC') #可以在tensorborad看到图片，add_image函数需要三个参数
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x",3*i,i) #第一个参数为名字，第二个参数为y轴，第三个参数为x轴
writer.close()