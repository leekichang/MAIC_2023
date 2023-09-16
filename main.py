from models import *
from dataset import *
import config as cfg

if __name__ == '__main__':
    model = CNN()
    model.to('cuda')
    dataset = MAIC()
    dataloader = DataLoader(dataset, 32, shuffle=False, drop_last=False)
    
    for epoch in range(100):
        for data in tqdm(dataloader):
            x = data['data']
            x = x.to('cuda')
            sex = data['gender']
            target = data['age']
            pred = model(x)
            
        