from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
img_path = "dataset/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs")
#1,使用transforms
tensor_trans = transforms.ToTensor() #创建一个tensor对象
tensor_img = tensor_trans(img) #把img变成tensor类型,tensor数据类型相当于包装了反向神经网络所需的所有参数，可以理解为这个类型就是神经网络特定数据类型
# print(tensor_img)
writer.add_image("Tensor_img",tensor_img)
writer.close()