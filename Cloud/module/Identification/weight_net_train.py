import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# plt.scatter(x,y)
# plt.scatter(x.data,y.data)
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(10,180)
        self.hidden2 = nn.Linear(180,360)
        self.hidden3 = nn.Linear(360,180)
        self.hidden4 = nn.Linear(180,2)
        self.predict = nn.Linear(2,1)

    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out = self.hidden3(out)
        out = F.relu(out)
        out = self.hidden4(out)
        out = F.relu(out)
        out =self.predict(out)

        return out

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
    #y = x.pow(3)+0.1*torch.randn(x.size())

    x = torch.tensor(torch.load("./userdata/kps.pt")).to(device)#
    y = torch.tensor(torch.load("./userdata/weight.pt")).to(device)#
    index=np.arange(y.size()[0])
    save_dir='./userdata/fusion_weight.pth'

    x , y =(Variable(x),Variable(y))

    net = Net().to(device)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(),lr = 0.0001)
    loss_func = torch.nn.MSELoss()

    plt.ion()
    plt.show()

    for e in range(10):
        for t in range(y.size()[0]):
            prediction = net(x[t])
            loss = loss_func(prediction,y[t])
            if t==0:
                total_loss=loss.data
            else:
                total_loss=total_loss+loss.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t%1000 ==0:
                #plt.cla()
                #plt.scatter(index, y.data.numpy())
                #plt.plot(index, prediction.data.numpy(), 'r-', lw=5)
                #plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
                #plt.pause(0.05)
                print('epoch:',e,'step:',t,'predict:',prediction.data,'loss:',loss.data)

        average_loss=total_loss/t
        print('epoch:',e,'loss:',average_loss)

        #if loss.data<=0.0002:
            #break

    state = {'net':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':t}
    torch.save(state, save_dir)

    #plt.ioff()
    #plt.show()
