import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm

import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out
    
class SAE(nn.Module):
    def __init__(self, in_channels, dec_channels, latent_size, private_classes):
        super(SAE, self).__init__()
        self.dec_channels = dec_channels
        self.e_conv1 = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels, dec_channels,
                                               kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        # 64x64
        self.e_conv2 = nn.Sequential(SpectralNorm(nn.Conv2d(dec_channels, dec_channels*2,
                                               kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        #32x32
        self.e_conv3 = nn.Sequential(SpectralNorm(nn.Conv2d(dec_channels*2, dec_channels*4,
                                               kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        #16x16
        self.e_conv4 = nn.Sequential(SpectralNorm(nn.Conv2d(dec_channels*4, dec_channels*8,
                                               kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        #8x8
        self.e_conv5 = nn.Sequential(SpectralNorm(nn.Conv2d(dec_channels*8, dec_channels*16,
                                               kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        #4x4
        
        self.e_fc_1 = nn.Linear(dec_channels*16*4*4, latent_size)

        self.e_attn1 = Self_Attn(dec_channels*8, 'relu')
        self.e_attn2 = Self_Attn(dec_channels*16, 'relu')

        self.private_embedding = nn.Linear(latent_size, private_classes)
        
        self.private_latent = self.private_embedding.weight
        
        self.d_fc_1 = nn.Linear(latent_size*2, dec_channels*16*4*4)

        self.d_conv1 = nn.Sequential(SpectralNorm(nn.ConvTranspose2d(dec_channels*16, dec_channels*8,
                                                kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        
        self.d_conv2 = nn.Sequential(SpectralNorm(nn.ConvTranspose2d(dec_channels*8, dec_channels*4,
                                                kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        
        self.d_conv3 = nn.Sequential(SpectralNorm(nn.ConvTranspose2d(dec_channels*4, dec_channels*2,
                                                kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        
        self.d_conv4 = nn.Sequential(SpectralNorm(nn.ConvTranspose2d(dec_channels*2, dec_channels,
                                                kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        
        self.d_conv5 = nn.Sequential(SpectralNorm(nn.ConvTranspose2d(dec_channels, in_channels,
                                                kernel_size=(4,4), stride=2, padding=1)),
                                    nn.Tanh())

        self.d_attn1 = Self_Attn(dec_channels*2, 'relu')
        self.d_attn2 = Self_Attn(dec_channels,  'relu')

    def encoder(self, x):
        x = self.e_conv1(x)
        x = self.e_conv2(x)
        x = self.e_conv3(x)
        x = self.e_conv4(x)
        x = self.e_attn1(x)
        x = self.e_conv5(x)
        x = self.e_attn2(x)
        x = x.view((-1, self.dec_channels*16*4*4))
        
        z = self.e_fc_1(x)
        return z
    
    def decoder(self, z, y, is_private):
        bs = z.size(0)
        if is_private:
            z_priv = self.private_latent.mean(dim=0).squeeze(0).repeat((bs,1))
        else:
            z_priv = self.private_latent[[y]].view((bs,-1))
        recon = torch.cat([z, z_priv], 1)
        recon = self.d_fc_1(recon)
        recon =recon.view(-1,self.dec_channels*16, 4, 4)
        recon = self.d_conv1(recon)
        recon = self.d_conv2(recon)
        recon = self.d_conv3(recon)
        recon = self.d_attn1(recon)
        recon = self.d_conv4(recon)
        recon = self.d_attn2(recon)
        recon = self.d_conv5(recon)
        return recon
    
    def forward(self, x, y, is_private=False):
        latent = self.encoder(x)
        recon = self.decoder(latent, y, is_private)
        return latent, recon
    
class DisNet(nn.Module):
    def __init__(self, latent_size, hidden_channels, private_classes):
        super(DisNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(latent_size, hidden_channels*16),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(hidden_channels*16))
        
        self.fc2 = nn.Sequential(nn.Linear(hidden_channels*16, hidden_channels*8),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(hidden_channels*8))
        
        self.fc3 = nn.Linear(hidden_channels*8, private_classes)
                                 
        
    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        return self.fc3(y)
    
# enc = SAE(3, 32, 1000, 307)
# x = torch.randn((10,3,128))
# z = enc.encoder(x)
# disc = DisNet(1000, 32,307)