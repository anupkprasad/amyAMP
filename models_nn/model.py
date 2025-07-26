from torch import nn
import torch
import torch.autograd as autograd


"""
# Weights initializer
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def init_weights(Layer):
    name = Layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(Layer.weight, mean=0, std=0.02)
        if Layer.bias is not None:
            torch.nn.init.constant_(Layer.bias, 0)

class TransposedCBR(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.model(x)


class CRBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding, bias=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, padding, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_num = 64):
        super().__init__()
        self.model = nn.Sequential(
            CRBlock(1, hidden_num, (1, in_dim), (1, 1), (0, 0), bias=False),
            CRBlock(hidden_num, hidden_num * 2, (4, 1), (2, 1), (1, 0), bias=False),
            CRBlock(hidden_num * 2, hidden_num * 4, (8, 1), (1, 1), (1, 0), bias=False),
            CRBlock(hidden_num * 4, hidden_num * 8, (4, 1), (2, 1), (1, 0), bias=False),
            nn.Conv2d(hidden_num * 8, 100, (5, 1), (1, 1), (0, 0), bias=False),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_dim, hidden_num = 64, out_dim = 6):
        super().__init__()
        self.model = nn.Sequential(
            TransposedCBR(in_dim, hidden_num * 8, (5, 1), (1, 1), (0, 0)),
            TransposedCBR(hidden_num * 8, hidden_num * 4, (4, 1), (2, 1), (1, 0)),
            TransposedCBR(hidden_num * 4, hidden_num * 2, (8, 1), (1, 1), (1, 0)),
            TransposedCBR(hidden_num * 2, hidden_num, (4, 1), (2, 1), (1, 0)),
            nn.ConvTranspose2d(hidden_num, 1, (1, out_dim), (1, 1), (0, 0), bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


"""
# Discriminator
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(180 + 100, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1)
        )
    def forward(self, X, z):
        X = X.reshape(-1, 30*6) ## reshape single PC6 encoded peptide
        z = z.reshape(-1, 100)  ## reshape latent feature of single pep
        Xz = torch.cat([X, z], dim = 1)
        return self.layers(Xz)

"""
# Discriminator loss
"""
def D_loss(DG, DE, eps=1e-6):
    loss = torch.log(DE + eps) + torch.log(1 - DG + eps)    
    return -torch.mean(loss)

"""
# Encoder - Generator loss
"""
def EG_loss(DG, DE, eps=1e-6):
    loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
    return -torch.mean(loss)

"""
WGAN-GP loss
"""
class loss_GP():
    def __init__(self, C, x, z_hat, x_tilde, z):
        self.C = C
        self.x = x
        self.z_hat = z_hat
        self.x_tilde = x_tilde
        self.z = z
    
    def criticize(self):
      x, z_hat, x_tilde, z = self.x, self.z_hat,self.x_tilde,self.z  
        
      input_x = torch.cat((x, x_tilde), dim=0)
      input_z = torch.cat((z_hat, z), dim=0)
      output = self.C(input_x, input_z)
      data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
      return data_preds, sample_preds

    def calculate_grad_penalty(self):
      x, z_hat, x_tilde, z =  self.x, self.z_hat,self.x_tilde,self.z  

      bsize = x.size(0)
      eps = torch.rand(bsize, 1, 1, 1).to(x.device) # eps ~ Unif[0, 1]
      intp_x = eps * x + (1 - eps) * x_tilde
      intp_z = eps * z_hat + (1 - eps) * z
      # check following two lines
      # intp_x.requires_grad = True
      # intp_z.requires_grad = True
      C_intp_loss = self.C(intp_x, intp_z).sum()
      grads = autograd.grad(C_intp_loss, (intp_x, intp_z), retain_graph=True, create_graph=True)
      grads_x, grads_z = grads[0].view(bsize, -1), grads[1].view(bsize, -1)
      grads = torch.cat((grads_x, grads_z), dim=1)
      grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
      return grad_penalty
    
    def loss(self,lamb=10):
      data_preds, sample_preds = self.criticize()
      EG_loss = torch.mean(data_preds - sample_preds)
      C_loss = -EG_loss + lamb * self.calculate_grad_penalty()
      E_loss = torch.mean(data_preds)
      G_loss = torch.mean(sample_preds)
      return C_loss, EG_loss, E_loss, G_loss
    

def get_model_and_optimizer(latent_size = 100):
    encoded_num = 6
    hidden_size = 64
    generator = Generator(latent_size, hidden_size, encoded_num)
    encoder = Encoder(encoded_num, hidden_size)
    discriminator = Discriminator()
    return generator, encoder, discriminator
