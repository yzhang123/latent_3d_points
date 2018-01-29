import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from data_loader import DataLoader
import torch.optim as optim
import os
import pdb


class D(nn.Module):
    def __init__(self, in_dim=128, out_dim=1):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, 256), 
            nn.ELU(inplace=True),
            nn.Linear(256, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        output = self.main(x)
        return output
    
class G(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ELU(inplace=True),
            nn.Linear(128, 128)
        )
    def forward(self, x):
        output = self.main(x)
        return output
    
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

def generate_noise(var):
    var.data.normal_(mean=0, std=0.2)
    
    
if __name__=='__main__':
    data_file = '../data/plane_hidden.npy'
    num_epochs=300
    batch_size=50
    x_dim=128
    z_dim=128
    save_dir = '../data/lgan_plane'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    d = D()
    g = G()
    # initializing weights
    for x in [d, g]:
        x.cuda()
        x.apply(weights_init)
       
    optimD = optim.Adam([{'params': d.parameters()}], lr=0.0001, betas=(0.5, 0.999))
    optimG = optim.Adam([{'params': g.parameters()}], lr=0.0001, betas=(0.5, 0.999))
    criterionD = nn.BCEWithLogitsLoss().cuda()
    criterionG = nn.BCEWithLogitsLoss().cuda()
    
    real = Variable(torch.FloatTensor(batch_size, x_dim)).cuda()
    noise = Variable(torch.FloatTensor(batch_size, z_dim)).cuda()
    label = Variable(torch.FloatTensor(batch_size, 1)).cuda()
    
        
    for epoch in xrange(num_epochs):
        data_loader = DataLoader(data_file, batch_size=batch_size, shuffle=True, repeat=False).iterator()
        for batch_num in xrange(80):
            # train D
            for _ in xrange(1):
#                 pdb.set_trace()
                optimD.zero_grad()
                batch = data_loader.next()
                x = torch.from_numpy(batch)
                real.data.copy_(x)
                logit_real = d(real)
                label.data.fill_(1)
                loss_real = criterionD(logit_real, label)
                loss_real.backward()

                generate_noise(noise)
                x_fake = g(noise)
                logit_fake = d(x_fake.detach())
                label.data.fill_(0)
                loss_fake = criterionD(logit_fake, label)
                loss_fake.backward()

                loss = loss_real + loss_fake

                optimD.step()

            
            # Train G
            optimG.zero_grad()
            generate_noise(noise)
            x_fake = g(noise)
            logit_fake = d(x_fake)
            label.data.fill_(1)
            loss_fake = criterionG(logit_fake, label)
            loss_fake.backward()
            optimG.step()
            
            d_loss = loss.data.cpu().numpy().mean()
            g_loss = loss_fake.data.cpu().numpy().mean()
            
            if batch_num % 10 == 0:
                print('epoch: {0}, iter: {1}, d_loss: {2}, g_loss: {3}'.format(epoch, batch_num, d_loss, g_loss))
            
            
        torch.save(g, save_dir + '/G_network_{0}.pth'.format(epoch))
            
            
            
        

    
            
