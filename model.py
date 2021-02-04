import torch.nn as nn
import torch
from torchvision import models
#model.py
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat, batch_norm = True)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True, batch_norm = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16_bn(pretrained = True)
            self._initialize_weights()
            model_keys = list(self.frontend.state_dict().keys())
            model_dict = {}
            for k in range(len(self.frontend.state_dict().keys())):
                dic1 = {model_keys[k]: list(mod.state_dict().values())[k]}
                model_dict.update(dic1)
            self.frontend.load_state_dict(model_dict)
            
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                


class Dense(nn.Module):
    def __init__(self, load_weights=False):
        super(Dense, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        #self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat, batch_norm = True)
        self.block1=nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=1,padding=1,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))
            #nn.Upsample(size=,scale_factor=None,mode='nearest',align_corners=None))
        self.block2=nn.Sequential(
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            nn.Conv2d(576,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=2,padding=2,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))        
        self.block3=nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=3,padding=3,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))
        self.block4=nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640,512,kernel_size=3,stride=1,dilation=1,padding=1,bias=False))
        
        #nn.ReLU(inplace=True)
        #self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
         #DDCB2
        self.block12=nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=1,padding=1,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))
            #nn.Upsample(size=,scale_factor=None,mode='nearest',align_corners=None))
        self.block22=nn.Sequential(
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            nn.Conv2d(576,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=2,padding=2,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))        
        self.block32=nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=3,padding=3,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
           # nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))
        self.block42=nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640,512,kernel_size=3,stride=1,dilation=1,padding=1,bias=False))
        #DDCB3
        self.block13=nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=1,padding=1,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))
            #nn.Upsample(size=,scale_factor=None,mode='nearest',align_corners=None))
        self.block23=nn.Sequential(
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            nn.Conv2d(576,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=2,padding=2,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))        
        self.block33=nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=3,padding=3,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
           # nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))
        self.block43=nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640,512,kernel_size=3,stride=1,dilation=1,padding=1,bias=False))
        #DDCB4
        self.block14=nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=1,padding=1,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))
            #nn.Upsample(size=,scale_factor=None,mode='nearest',align_corners=None))
        self.block24=nn.Sequential(
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            nn.Conv2d(576,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=2,padding=2,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))        
        self.block34=nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640,256,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,stride=1,dilation=3,padding=3,bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
           # nn.Conv2d(64,512,kernel_size=1,stride=1,dilation=1,bias=False),
            nn.Dropout2d(p=0.1))
        self.block44=nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640,512,kernel_size=3,stride=1,dilation=1,padding=1,bias=False))
        #output
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=3,stride=1,dilation=1,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3,stride=1,dilation=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, kernel_size=1,stride=1,dilation=1,bias=False))
        if not load_weights:
            mod = models.vgg16_bn(pretrained = True)
            self._initialize_weights()
            model_keys = list(self.frontend.state_dict().keys())
            model_dict = {}
            for k in range(len(self.frontend.state_dict().keys())):
                dic1 = {model_keys[k]: list(mod.state_dict().values())[k]}
                model_dict.update(dic1)
            self.frontend.load_state_dict(model_dict)
            
            '''self._initialize_weights()
            model_dict = self.state_dict()
            pretrained_dict1 = torch.load('../working2v2/tensor(8.8255, device=\'cuda:0\')model_best.pth.tar')
            pretrained_dict = pretrained_dict1['state_dict']
            #print('***********')
            #print(pretrained_dict.items())
            #print('***********')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            #print(pretrained_dict)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)'''
    def forward(self,_input):
        front0=self.frontend(_input)
        #print(front0.size())
        #print(front0)
        
        block1=self.block1(front0)
        front1=torch.cat((front0,block1),dim=1)
        #print(front1.size())

        block2=self.block2(front1)
        front2=torch.cat((front0,block2),dim=1)
        #print(front2.size())

        front2=torch.cat((front2,block1),dim=1)
        #print(front.size())

        block3=self.block3(front2)
        front3=torch.cat((front0,block3),dim=1)
        #print(front3.size())
        front3=torch.cat((block2,front3),dim=1)
        
        block4=self.block4(front3)
        front4=block4
         #DDCB2
        block1=self.block12(front4)
        front1=torch.cat((front4,block1),dim=1)
        #print(front1.size())

        block2=self.block22(front1)
        front2=torch.cat((front4,block2),dim=1)
        #print(front2.size())

        front2=torch.cat((front2,block1),dim=1)
        #print(front.size())

        block3=self.block32(front2)
        front3=torch.cat((front4,block3),dim=1)
        #print(front3.size())
        front3=torch.cat((block2,front3),dim=1)
        
        block42=self.block42(front3)
        front42=block42+front0
        #DDCB3
        block1=self.block13(front42)
        front1=torch.cat((front42,block1),dim=1)
        #print(front1.size())

        block2=self.block23(front1)
        front2=torch.cat((front42,block2),dim=1)
        #print(front2.size())

        front2=torch.cat((front2,block1),dim=1)
        #print(front.size())

        block3=self.block33(front2)
        front3=torch.cat((front42,block3),dim=1)
        #print(front3.size())
        front3=torch.cat((block2,front3),dim=1)
        
        block43=self.block43(front3)
        front43=block43
        #DDCB4
        block1=self.block14(front43)
        front1=torch.cat((front43,block1),dim=1)
        #print(front1.size())

        block2=self.block24(front1)
        front2=torch.cat((front43,block2),dim=1)
        #print(front2.size())

        front2=torch.cat((front2,block1),dim=1)
        #print(front.size())mask

        block3=self.block34(front2)
        front3=torch.cat((front43,block3),dim=1)
        #print(front3.size())
        front3=torch.cat((block2,front3),dim=1)
        
        block44=self.block44(front3)
        front44=block44
        out = self.output_layer(front44)
        return out
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
