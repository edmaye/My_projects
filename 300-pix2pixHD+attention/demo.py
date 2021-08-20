from src.pix2pixHD.models.networks import GlobalGenerator,get_norm_layer
import torch



norm_layer = get_norm_layer(norm_type='instance')     
    
netG = GlobalGenerator(3, 3, 64, 4, 9, norm_layer) 
x = torch.zeros((4,3,512,512))
out = netG(x)
print(out.shape)