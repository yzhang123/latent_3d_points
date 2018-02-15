import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
from data_loader import DataLoader
import torch.optim as optim
import os
import os.path as osp

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

def calc_gradient_penalty(real_data, fake_data, batch_size, d):
        #pdb.set_trace()
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = d(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 0.01
        return gradient_penalty
    
if __name__=='__main__':
    import sys
    
    data_file = sys.argv[1] #'../data/plane_hidden.npy'
    save_dir = sys.argv[2] #'../data/lgan_plane' will create this


    num_epochs=1000
    batch_size=50
    x_dim=128
    z_dim=128
    critic_steps = 5
    lambda_grad_penalty = 0.1
    weight_clip = 1
    
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

    
    real = Variable(torch.FloatTensor(batch_size, x_dim), requires_grad=False).cuda()
    noise = Variable(torch.FloatTensor(batch_size, z_dim), requires_grad=False).cuda()
    label_zero = Variable(torch.FloatTensor(batch_size, 1).fill_(0), requires_grad=False).cuda()
    label_one = Variable(torch.FloatTensor(batch_size, 1).fill_(1), requires_grad=False).cuda()
    
    one = torch.cuda.FloatTensor([1])
    mone = one * -1
    
    # pdb.set_trace()
    f = open(osp.join(save_dir, 'log.txt'), 'a')    
    for epoch in xrange(num_epochs):
        data_loader = DataLoader(data_file, batch_size=batch_size, shuffle=True, repeat=False).iterator()
        batch_num = 0
        while True:
            try:
                D_loss = np.zeros(critic_steps)
                # train D
                for p in d.parameters():
                    p.requires_grad = True
                for j in xrange(critic_steps):
                    D_loss = np.zeros(critic_steps)
                    for p in d.parameters():
                        p.data.clamp_(-1 * weight_clip, weight_clip)


                    optimD.zero_grad()
                    batch = data_loader.next()
                    x = torch.from_numpy(batch)
                    real.data.copy_(x)
                    errD_real = d(real).mean(0)

                    # fake part
                    generate_noise(noise)
                    fake_x = g(noise)
                    errD_fake = d(fake_x.detach()).mean(0)

                    errD = errD_fake - errD_real
                    errD.backward()

                    # train with gradient penalty
                    # gradient_penalty = calc_gradient_penalty(real.data, fake_x.data, batch_size=batch_size, d=d)
                    # weighted_grad_penalty = lambda_grad_penalty * gradient_penalty
                    
                    # # weighted_grad_penalty.backward()

                    # total_errD = errD + weighted_grad_penalty
                    # D_loss[j] = total_errD.data.cpu().numpy()


                    D_loss[j] = errD.data.cpu().numpy()
                    optimD.step()

                
   
                # Train G

                for p in d.parameters():
                    p.requires_grad = False
                optimG.zero_grad()

                generate_noise(noise)
                x_fake = g(noise)
                errG = -d(x_fake).mean(0)
                errG.backward()
                optimG.step()
                
                d_loss = D_loss.mean()
                d_loss_real = np.asscalar(errD_real.data.cpu().numpy())
                d_loss_fake = np.asscalar(errD_fake.data.cpu().numpy())
                g_loss = np.asscalar(errG.data.cpu().numpy())
                
                if batch_num % 1 == 0:
                    print('epoch: {0}, iter: {1}, d_loss: {2}, g_loss: {3}, d_loss_real: {4}, d_loss_fake: {5}'.format(epoch, batch_num, d_loss, g_loss, d_loss_real, d_loss_fake))
                    f.write('epoch: {0}, iter: {1}, d_loss: {2}, g_loss: {3}, d_loss_real: {4}, d_loss_fake: {5} \n'.format(epoch, batch_num, d_loss, g_loss, d_loss_real, d_loss_fake))
                batch_num += 1
            except:
                break
            
        torch.save(g, save_dir + '/G_network_{0}.pth'.format(epoch))
            
            
            

    
            
