import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

#gerekli yapay zeka kütüphalenerimizi import ediyoruz


#resimleri uygun hale getiriyorum

#tüm resimlere sırayla uygulanacak olan işlemleri pipeline oluşturma
donusturme=transforms.Compose([ 
    transforms.Resize((224,224)), # bütün resimlerin boyutlarını eşit hale getiriyoruz
    transforms.ToTensor(), # resimleri pikselden tensora ceviriyoruz 0-1 muhabbeti 
    transforms.Normalize([0.485, 0.456,0.406],
                         [0.229,0.224,0.225])
                         #modelin daha stabil çalışması için gerekli kod yapay zekadan aldım 
])

egitim_dosyasi='data/train' # verisetimizin bulundugu klasör
deger_dosyasi='data/val'    # veri setinin küçük parçası (ilk egitim olarak bunu kullanacaz daha hızlı egitim olması adına)

egitim_ayarlari=datasets.ImageFolder(egitim_dosyasi,transform=donusturme)
deger_ayarlari=datasets.ImageFolder(deger_dosyasi,transform=donusturme)
# datasets.ımagefolder klasor isimlerini sinif ismi alarak hepsini tek tek okuyup istenilen formata transfor eder
#  yani bizim donusturme değişkenimizdeki işlemi yapar.

egitim_yukleyici=DataLoader(egitim_ayarlari,batch_size=32,shuffle=True)
deger_yukleyici=DataLoader(deger_ayarlari,batch_size=32,shuffle=True)

# dataloader veri setini parcalara böler yani batchlere 32 lik parcalar halinde egitimi uygun gördüm
#suffle true ise ezber yapmaması için gerekli kod


sinif_isimleri=egitim_ayarlari.classes
print('sınıflar : ',sinif_isimleri)


#************MODEL TARAFI***************

class ResBlock(nn.Module):
    def __init__(self, giris, cikis,stride=1):
        super().__init__()
        self.c1=nn.Conv2d(giris,cikis,3,stride,1,bias=False)
        self.b1=nn.BatchNorm2d(cikis)
        self.c2=nn.Conv2d(cikis,cikis,3,1,1,bias=False)
        self.b2=nn.BatchNorm2d(cikis)



        self.short=nn.Sequential()
        if stride !=1 or giris !=cikis:
            self.short=nn.Sequential(
                nn.Conv2d(giris,cikis,1,stride,bias=False),
                nn.BatchNorm2d(cikis)
            )

    def forward(self,x):
        y=torch.relu(self.b1(self.c1(x)))
        y=self.b2(self.c2(y))
        y+=self.short(x)
        y=torch.relu(y)
        return y 


class MyNet(nn.Module):
    def __init__(self, sinif_num):
        super().__init__()
        self.katman0 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.l0=nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )



        self.katman1=ResBlock(64,64)
        self.katman2=ResBlock(64,128,stride=2)
        self.katman3=ResBlock(128,256,stride=2)
        self.katman4=ResBlock(256,512,stride=2)
        self.ortalama=nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, sinif_num)

    def forward(self,x):

        x=self.katman0(x)
        x=self.katman1(x)
        x=self.katman2(x)
        x=self.katman3(x)
        x=self.katman4(x)
        x=self.ortalama(x)
        x=torch.flatten(x,1)
        x=self.fc(x)
        return(x)
    


# egitim kısmı


device = "cuda" if torch.cuda.is_available() else "cpu"
#bilgisayardaki cpu yu kullanmasını söyluyoruz egitim daha hızlı gerçekleşiyor 

kullanilacak_model=MyNet(len(sinif_isimleri)).to(device)
# kac sıfınımız oldugunu belirtiyoruz .to(device)  modeli cpu ya taşıyoruz .

loss_fonksiyonu=nn.CrossEntropyLoss()

opt=optim.Adam(kullanilacak_model.parameters(),lr=0.0001)
#optimizer olarak adamı sectık  lr öğrenme oranını belirttik 


def train_one_epoch(ep):
    kullanilacak_model.train()
    toplam=0
    for i ,(resim,etiket) in enumerate(egitim_yukleyici):
        resim,etiket =resim.to(device),etiket.to(device)

        opt.zero_grad()
        cikiss=kullanilacak_model(resim)
        loss=loss_fonksiyonu(cikiss,etiket)
        loss.backward()
        opt.step()

        toplam+=loss.item()

        if i % 100 ==0:
            print("epoch:", ep, "| batch:", i, "| loss:", float(loss))
    print("epoch", ep, "bitti ortalama loss=", toplam / len(egitim_yukleyici))




if __name__ == "__main__":
    for e in range(1, 6):
        train_one_epoch(e)

    torch.save(kullanilacak_model.state_dict(), "model.pth")
    print("Model kaydedildi.")
