# from os import listdir
# import shutil
# from os.path import isfile, join
# img_path = "/home/jamal/Desktop/LFW attributes/lfw/"
# save_path = "/home/jamal/Desktop/LFW attributes/more_than_once/"
# onlyfiles = [f for f in listdir(img_path)]
# 
# for i in range(len(onlyfiles)):
#     tem_path = img_path + onlyfiles[i]
#     if len(listdir(tem_path))>1:
#         s_path = save_path + onlyfiles[i]
#         shutil.copytree(tem_path,s_path)
#     print(i/len(onlyfiles))
# 
#         
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
from skimage import io, transform
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import sklearn.metrics as mt

# additional files is located in /home/jamal/Downloads/additional_files/
def save_checkpoint(state, filename='/home/jamal/Downloads/additional_files/checkpoint.pth.tar'):
    torch.save(state, filename)
    
file_path ='/home/jamal/Downloads/additional_files/umdfaces_females_batch2.csv'

images_path = '/home/jamal/Downloads/umdfaces_batch2/'

load_net = 1
# columns = ['ImgId','5_o_Clock_Shadow', ' Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

columns=['SUBJECT_ID','FILE','FACE_X','FACE_Y','FACE_H','FACE_W','PR_MALE','PR_FEMALE']

cele_attrib = pd.read_csv(file_path,delimiter = ",",names = columns)
# lfw = cele_attrib.set_index('SUBJECT_ID')
lfw = cele_attrib

#len_attrib = len(cele_attrib)
# Select random images form celeba dataset
# rnd_set = np.random.permutation(len_attrib)[0:5]
# for i in rnd_set:
#     idx = ("{:06d}.png".format(i))
#     img_path = images_path+idx
#     img = plt.imread(img_path)
#     plt.imshow(img)
#     plt.show()
#     print(cele_attrib['Male'][i-1])
#     
import torch
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv11 = nn.Conv2d(3,64,5)
        self.n11 = nn.BatchNorm2d(64)
        self.conv21 = nn.Conv2d(64,96,5)
        self.n21 = nn.BatchNorm2d(96)
        self.conv31 = nn.Conv2d(96,128,3)
        self.n31 = nn.BatchNorm2d(128)
        
        self.conv12 = nn.Conv2d(3,32,7)
        self.n12 = nn.BatchNorm2d(32)
        self.conv22 = nn.Conv2d(32,75,3)
        self.n22 = nn.BatchNorm2d(75)
        #self.conv32 = nn.Conv2d(75,100,3)
        #self.n32 = nn.BatchNorm2d(100)
        
        
        self.ip1 = nn.Linear(1712,128)   # change the first parameter in case you change the size of the small image
        self.ip2 = nn.Linear(128,2)
        self.ip3 = nn.Linear(1712,128,False)   # change the first parameter in case you change the size of the small image
        self.ip4 = nn.Linear(128,2,False)
        
    
    def forward(self,x,y):
        x = self.conv11(x)
        x = self.n11(x)
        x = F.relu(x)
        
        x = F.max_pool2d(x,5,stride = 3)
        
        
        x = self.conv21(x)
        
        x = self.n21(x)
        x = F.relu(x)
        x = F.max_pool2d(x,5,stride = 3)
        x = self.conv31(x)
        
        x = self.n31(x)
        x = F.relu(x)
        x = F.max_pool2d(x,5)
        x = x.view(-1,512)
        
        
        y = self.conv12(y)
        
        y = self.n12(y)
        y = F.relu(y)
        # y = F.max_pool2d(y,5,stride = 1)
        y = self.conv22(y)
        y = self.n22(y)
        y = F.relu(y)
        # y = F.max_pool2d(y,5,stride = 2)
        
        #y = self.conv32(y)
        #y = F.max_pool2d(y,3)
        #y = self.n32(y)
        #y = F.relu(y)
        y = y.view(-1,1200)  # change the second parameter in case you change the size of the small image
        
        x = torch.cat((x,y),1)
        x1 = self.ip1(x)
        x1 = F.relu(x1)
        x1 = self.ip2(x1)
        x2 = self.ip3(x)
        x2 = F.relu(x2)
        x2 = self.ip4(x2)
        # x2 = x2.mul(-1)
        # x = F.relu(x)
        # x = F.softmax(x,1)
        return x1, x2
      
class Male_Female_dataset(Dataset):
    
    def __init__(self,root_dir,shape,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.shape = shape
        self.images_names = os.listdir(images_path)
        
    def __len__(self):
        return len(cele_attrib)
        
    def __getitem__(self,idx):
        t = torch.rand(1);
        lst = cele_attrib.iloc[idx].tolist()
        img1_name = os.path.join(self.root_dir,lst[0])
        image1 = io.imread(img1_name)
        image1 = image1[int(lst[2]):int(lst[2]+lst[4]),int(lst[1]):int(lst[1]+lst[3]),:]
        # import cv2
        # cv2.imshow('a',img)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        image1 = transform.resize(image1,(150,150))
       
        
        if t > 0.5: #different
            different = False
            while not(different):
                h = torch.randint(0,len(self),(1,1))
                ht = torch.Tensor.numpy(h)
                if lfw.iloc[idx][0] != lfw.iloc[int(ht[0])][0]:
                    different = True
            
            lstd = cele_attrib.iloc[int(ht[0])].tolist()        
                
            img2_name = os.path.join(self.root_dir,lstd[0])
            image2 = io.imread(img2_name)
            image2 = image2[int(lstd[2]):int(lstd[2]+lstd[4]),int(lstd[1]):int(lstd[1]+lstd[3]),:]
            # import cv2
            # cv2.imshow('a',image2)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
            image2 = transform.resize(image2,self.shape)
            image2 = torch.Tensor.float(torch.from_numpy(image2))
            image2 = (torch.Tensor.permute(image2,(2,0,1)))
            image2 = image2/256.0
            image1 = torch.Tensor.float(torch.from_numpy(image1))
            image1 = (torch.Tensor.permute(image1,(2,0,1)))
            image1 = image1/256.0
            annot = 0
            gender =  1 if lst[-2]>.5 else -1

            
        else: #same
            id = cele_attrib.iloc[idx].tolist()[0]
            idx_new = idx+1
            if  cele_attrib.iloc[idx+1].tolist()[0]==id:
                lstd = cele_attrib.iloc[idx+1].tolist()
                sta = os.path.join(self.root_dir,lstd[0])
            else:
                lstd = cele_attrib.iloc[idx-1].tolist()
                sta = os.path.join(self.root_dir,lstd[0])
            # st = img1_name.split('/')
            # st = st[-1]
            # st = st.split('_')
            # nn = st[0:-1]
            # name = ' '.join(nn)
            # num = int(st[-1].split('.')[0])
            
            # cc = lfw.loc[name]
            # if isinstance(cc.imagenum,str):
            #     print(cc)
            # ls = cc.imagenum.tolist()
            # if str(num) in ls:
            #     ls.remove(str(num))
            # 
            # h = torch.randint(0,len(ls),(1,1))
            # ht = torch.Tensor.numpy(h)
            # ht = ht[0]
            # num2 = str(int(ls[int(ht)])).zfill(4)
            # sta = img1_name.split('_')
            # stemp = sta[-1].split('.')
            # stemp[0] = num2
            # stemp = '.'.join(stemp)
            # sta[-1]=stemp
            # sta = '_'.join(sta)
            image2 = io.imread(sta)
            image2 = transform.resize(image2,self.shape)
            image2 = torch.Tensor.float(torch.from_numpy(image2))
            image2 = (torch.Tensor.permute(image2,(2,0,1)))
            image2 = image2/256.0
            image1 = torch.Tensor.float(torch.from_numpy(image1))
            image1 = (torch.Tensor.permute(image1,(2,0,1)))
            image1 = image1/256.0
            annot = 1
            gender = 1 if lst[-2]>.5 else -1
        # annot = annot.astype('float').reshape(-1,2)
        sample = {'image1':image1,'image2':image2, 'same' : annot, 'gender' : gender}
        return sample

use_cuda = torch.cuda.is_available()
use_cuda=0
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
shape = (12,12)
ip_shape = (150,150)
train_ratio = 0
val_ratio = 0
test_ratio = 1-train_ratio-val_ratio
dataset = Male_Female_dataset(images_path,shape)
# dataset.__getitem__(175534-1)
if load_net:
    net = Net().to(device)
    checkpoint = torch.load('/home/jamal/Desktop/trained nn/checkpoint_0.001.pth.tar',map_location='cpu')
    net.load_state_dict ( checkpoint['state_dict'])
    optimizer = optim.Adam(net.parameters(),lr = 0.001, weight_decay = 0.0005)
    optimizer.load_state_dict = checkpoint['optimizer']
else:
    
    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(),lr = 0.001, weight_decay = 0.0005)
    checkpoint= {'epoch':0}
    

optimizer = optim.Adam(net.parameters(),lr = 0.01, weight_decay = 0.0005)
index = np.random.permutation(len(dataset)-5)
train_data_length = int(train_ratio*len(index))
val_data_length = int(val_ratio*len(index))
test_data_length = int(test_ratio*len(index))
train_index = index[:train_data_length]
val_index = index[train_data_length:(train_data_length+val_data_length)]
test_index = index[train_data_length+val_data_length:]

train_dataloader = DataLoader(dataset,batch_size=50,sampler = SubsetRandomSampler(train_index))

err_same = []
err_gender = []

acc1 = []
acc2 = []

criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()

# for ep in range(checkpoint['epoch'],5):
#     for i,data in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         input1, input2 , label, gender = data.items()
#         input1, input2 = input1[1].to(device), input2[1].to(device)
#         label1 = label[1].to(device)
#         gender = gender[1].to(device)
#         gender = (gender+1)/2
#         #label = torch.Tensor.long(label[1])
#         output = net(input1,input2)
#         loss1 = criterion (output[0],label1)
#         loss1.backward(retain_graph=True)
#         # optimizer.step()
#         loss2 = criterion2 (output[1],gender)
#         loss2 = loss1 - loss2.mul(0.0001)  #new line 
#         loss2.backward()
#         optimizer.step()
#         err_same.append(loss1.item())
#         err_gender.append(loss2.item())
#         if (i%5==0):
#             print (loss1,i,ep)
#             print (loss2,i,ep)
#                     
#         if (i%50==0):
#             val_index1 = np.random.permutation(val_index)[:100]
#             val_dataloader = DataLoader(dataset,batch_size=30,sampler = SubsetRandomSampler(val_index1))
#             val_iter = iter(val_dataloader)
#     
#             total = 0
#             correct1 = 0
#             correct2 = 0
#             for j,dataj in enumerate(val_dataloader):
#                 input1j, input2j, labelj, gender = dataj.items()
#                 input1j, input2j = input1j[1].to(device), input2j[1].to(device)
#                 
#                 labelj = labelj[1].to(device)
#                 gender = gender[1].to(device)
#                 gender = (gender+1)/2
#                 output = net(input1j,input2j)
#                 _,predicted1 = torch.max(output[0].data,1)
#                 _,predicted2 = torch.max(output[1].data,1)
#                 total +=labelj.size(0)
#                 correct1 += (predicted1 == labelj).sum().item()
#                 correct2 += (predicted2 == gender).sum().item()
#             print('Accuracy_LH: %d %%'%(100*correct1/total))
#             print('Accuracy_MF: %d %%'%(100*correct2/total))
#             acc1.append(100*correct1/total) #low high
#             acc2.append(100*correct2/total) # male female
#     save_checkpoint({
#             'epoch': ep + 1,
#             'state_dict': net.state_dict(),
#             
#             'optimizer' : optimizer.state_dict(),
#         })
# #     
val_index1 = np.random.permutation(val_index)[:100]
val_dataloader = DataLoader(dataset,batch_size=30,sampler = SubsetRandomSampler(test_index))
val_iter = iter(val_dataloader)
predicted_lables = []
real_labels_g = []
predicted_lables_g = []
real_labels = []
total = 0
correct1 = 0
correct2 = 0
for j,dataj in enumerate(val_dataloader):
    input1j, input2j, labelj, gender = dataj.items()
    input1j, input2j = input1j[1].to(device), input2j[1].to(device)
    
    labelj = labelj[1].to(device)
    gender = gender[1].to(device)
    gender = (gender+1)/2
    output = net(input1j,input2j)
    _,predicted1 = torch.max(output[0].data,1)
    _,predicted2 = torch.max(output[1].data,1)
    total +=labelj.size(0)
    correct1 += (predicted1 == labelj).sum().item()
    correct2 += (predicted2 == gender).sum().item()
    predicted_lables.append(torch.Tensor.numpy(predicted1))
    real_labels.append(torch.Tensor.numpy(labelj))
    
    predicted_lables_g.append(torch.Tensor.numpy(predicted2))
    real_labels_g.append(torch.Tensor.numpy(gender))
    print('Accuracy_LH: %d %%'%(100*correct1/total))
    print('Accuracy_MF: %d %%'%(100*correct2/total))
    acc1.append(100*correct1/total)
    acc2.append(100*correct2/total)
    print(mt.confusion_matrix(np.array(real_labels).flatten(),np.array(predicted_lables).flatten()))
    
    print(mt.confusion_matrix(np.array(real_labels_g).flatten(),np.array(predicted_lables_g).flatten()))
# # for ep in range(checkpoint['epoch'],5):
#     for i,data in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         input1, input2 , label = data.items()
#         input1, input2 = input1[1].to(device), input2[1].to(device)
#         label1 = label[1].to(device)
#         #label = torch.Tensor.long(label[1])
#         output = net(input1,input2)
#         loss = criterion (output,label1)
#         loss.backward()
#         optimizer.step()
#         err.append(loss.item())
#         if (i%5==0):
#             print (loss,i,ep)
#         
#         if (i%50==0):
#             val_index1 = np.random.permutation(val_index)[:100]
#             val_dataloader = DataLoader(dataset,batch_size=30,sampler = SubsetRandomSampler(val_index1))
#             val_iter = iter(val_dataloader)
#     
#             total = 0
#             correct = 0
#             for j,dataj in enumerate(val_dataloader):
#                 input1j, input2j, labelj = dataj.items()
#                 input1j, input2j = input1j[1].to(device), input2j[1].to(device)
#                 
#                 labelj = labelj[1].to(device)
#                 output = net(input1j,input2j)
#                 _,predicted = torch.max(output.data,1)
#                 total +=labelj.size(0)
#                 correct += (predicted == labelj).sum().item()
#             print('Accuracy: %d %%'%(100*correct/total))
#             acc.append(100*correct/total)
#     save_checkpoint({
#             'epoch': ep + 1,
#             'state_dict': net.state_dict(),
#             
#             'optimizer' : optimizer.state_dict(),
#         })
#     
# 
#         
#         
#         
# 
#     
# # sample = dataset[1]
# # img = plt.imread(images_path+'000001.png')
# # img2 = np.expand_dims(img,0)
# # img3 = np.rollaxis(img2,3,1)
# # output = Net(torch.from_numpy(img3))