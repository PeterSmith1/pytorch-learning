import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#准备数据集

test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor())
#batch_size表示从数据集当中一次取几个数据集，shuffle属性设置一般为False，意为两次读取数据集顺序不一样，num_workers属性：加载数据进程数
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
#测试数据集第一张图片及target
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0
#drop_last设置为False,不舍弃不足batch_size一次性规定取的
# for data in test_loader:
#     imgs,targets = data
#     writer.add_images("test_imgs",imgs,step)
#     step = step + 1
for data in test_loader:
    imgs, targets = data
    writer.add_images("drop_last_imgs", imgs, step)
    step = step + 1
writer.close()