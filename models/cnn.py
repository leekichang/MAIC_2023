import torch
import torch.nn as nn

__all__ = [
    'CNN',
    ]

model_config={  
            'channel':[12, 4, 16, 32],
            'kernel' : 7,
            'stride' : 3,
            'linear' :[320,704],
            'groups' :1,
            'n_stage':1
        }

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups=1):
        super(ConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, groups=groups),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
    
    def forward(self, x):
        x = self.convs(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_class=1):
        super(CNN, self).__init__()
        self.config     = model_config
        self.convBlocks = nn.ModuleList([])
        for i in range(len(self.config['channel'])-1):
            self.convBlocks.append(ConvBlock(in_channel=self.config['channel'][i],
                                             out_channel=self.config['channel'][i+1],
                                             kernel_size=self.config['kernel'],
                                             stride=self.config['stride'],
                                             groups=self.config['groups']))
            
        self.drop_outs   = nn.ModuleList([nn.Dropout(0.1) for _ in range(3)])
        
        self.head        = nn.Linear(in_features=self.config['linear'][0], out_features=self.config['linear'][1])
        if num_class != 0:
            self.classifier  = nn.Linear(in_features=self.config['linear'][1]+1, out_features=num_class)
        else:
            self.classifier  = nn.Identity()
    
    def forward(self, x):
        
        x, sex = x[0], x[1]
        for idx in range(len(self.convBlocks)):
            x = self.drop_outs[idx](self.convBlocks[idx](x))
        sex = sex.reshape(x.size(0), 1)
        x = self.classifier(torch.cat((x.reshape(x.size(0), -1), sex), dim=1))
        return x
    
if __name__ == '__main__':
    model = CNN()
    x = torch.randn(2, 12, 5000)
    sex = torch.ones(2)
    out = model([x, sex])
    print(x.shape, out.shape)