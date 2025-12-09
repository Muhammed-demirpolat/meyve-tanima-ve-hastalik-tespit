import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
from PyQt6.QtGui import QPixmap
from PIL import Image
import torch
from torchvision import transforms


from egitim import MyNet, sinif_isimleri, device

# form sayfası

class yaprakApp(QWidget):
    def __init__(self,):
        super().__init__()
        self.setWindowTitle("Yaprak Hastalık Bulucu")
        self.setGeometry(200, 200, 400, 500)
        #sayfa başlığını belirliedik
        layout = QVBoxLayout()
        #resmi formda gösterdiğimiz kısım
        self.resim_etiketi=QLabel('Resim seçin')
        self.resim_etiketi.setFixedSize(350,350)
        #resmi ekrana sığdırma ilmemi 350x350 yaptık
        layout.addWidget(self.resim_etiketi)


        #hastalık tespit yazısı
        self.sonuc_etiketi=QLabel('tahmin burda yazacak')
        layout.addWidget(self.sonuc_etiketi)

        #resim secme butonu
        self.buton=QPushButton('resim Yükle')
        self.buton.clicked.connect(self.resim_sec)
        layout.addWidget(self.buton)

        self.setLayout(layout)

        #model cekme 
        self.model=MyNet(len(sinif_isimleri)).to(device)
        self.model.load_state_dict(torch.load('model.pth',map_location=device))
        self.model.eval()


        #resmi donuşturme ayarları egitim.py daki ile aynı
        self.cevirici = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])



    #resim secme fonsiyonu
    def resim_sec(self):
        dosya, _ = QFileDialog.getOpenFileName(self, "Resim Aç", "", "Resimler (*.jpg *.png *.jpeg)")
        if dosya:
            pix = QPixmap(dosya)
            self.resim_etiketi.setPixmap(pix.scaled(350, 350))

            # Tahmin kısmı
            img = Image.open(dosya).convert("RGB")
            img = self.cevirici(img).unsqueeze(0).to(device)

            with torch.no_grad():
                cikti = self.model(img)
                _, tahmin = torch.max(cikti, 1)

            self.sonuc_etiketi.setText("Tahmin: " + sinif_isimleri[tahmin.item()])


# Çalıştırma
if __name__ == "__main__":
    app = QApplication(sys.argv)
    pencere = yaprakApp()
    pencere.show()
    sys.exit(app.exec())