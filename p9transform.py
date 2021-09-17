from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
img_path = "data/train/ants_image/6240338_93729615ec.jpg"
img = Image.open(img_path)
# print(img)
writer = SummaryWriter("logs")
#1,使用transforms
tensor_trans = transforms.ToTensor() #创建一个tensor对象
tensor_img = tensor_trans(img) #把img变成tensor类型,tensor数据类型相当于包装了反向神经网络所需的所有参数，可以理解为这个类型就是神经网络特定数据类型
# print(tensor_img)
writer.add_image("Tensor_img",tensor_img)

#Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #两个参数分别为均值、标准差，让tensor_img归一化
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img) # img是PIL类型通过resize后变成 img_resize的PIL
img_resize = tensor_trans(img_resize) #把img_resize是PIL型变成tensor类型
writer.add_image("Resize",img_resize,0)
print(img_resize)

#Compose 等比例进行改变大小
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,tensor_trans]) #PIL -> PIL ->tensor
img_resize_2 = trans_compose(img)
writer.add_image("Compose",img_resize_2,1)

# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop(100)
trans_compose_2 = transforms.Compose([trans_random,tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()