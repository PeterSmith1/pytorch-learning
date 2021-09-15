from torch.utils.data.dataset import Dataset #torch.utils torch常用工具箱
from PIL import Image
import os
class MyData (Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir  #把root_dir 变为全局变量便于使用
        self.label_dir = label_dir #把label_dir 变为全局变量便于使用
        self.path = os.path.join(self.root_dir,self.label_dir) #获得图片的路径地址
        self.img_path = os.listdir(self.path) #获得图片列表地址
    def __getitem__(self, idx): #获取其中的每一个图片路径
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path) #获取长度
root_dir = "dataset/train"
ants_label_dir = "ants_image"
bees_label_dir = "bees_image"
ants_dataset = MyData(root_dir,ants_label_dir)  #创建数据集
bees_dataset = MyData(root_dir,bees_label_dir)
train_dataset = ants_dataset + bees_dataset
