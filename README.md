# 🍃 Leaf AI — Bitki Hastalığı + Meyve Tanıma Uygulaması

Leaf AI, bitki yapraklarından **hastalık tespiti** yapan ve aynı zamanda eklenen ek veri seti ile **meyve tanıma** (Muz, Elma, Mısır) özelliği bulunan bir PyTorch + PyQt6 projesidir.

Bu proje, tamamen sıfırdan oluşturulmuş bir Convolutional Neural Network (CNN) ve ResNet tarzı bloklar kullanılarak geliştirilmiştir.

---

## 🚀 Özellikler

* Bitki yapraklarında hastalık tanıma
* Eklenen meyve sınıfları:

  * **Muz (muz_saglikli)**
  * **Kırmızı Elma (kirmizi_elma)**
  * **Mısır (Corn)**
* PyQt6 arayüzü ile kolay kullanım
* Modeli eğitme ve eğitilmiş modeli kullanma dosyalarının ayrılması
* GPU destekli eğitim (CUDA varsa otomatik algılanır)

---

## 📁 Proje Yapısı

```
leaf_ai/
│
├── data/
│   ├── train/       # Eğitim veri seti klasörü
│   └── val/         # Opsiyonel doğrulama veri seti
│
├── src/
│   ├── train.py     # Modeli eğiten dosya
│   ├── app.py       # PyQt6 arayüzü ve model kullanım dosyası
│   └── model.pth    # Eğitilmiş model (otomatik oluşur)
│
└── README.md
```

---

## 🧠 Model Mimarisi

Proje, ResNet mantığına benzeyen özel bir ağ mimarisi kullanır:

* Residual bloklar (**ResBlock**) ile daha derin ve stabil model
* 5 katmanlı CNN yapısı
* Adaptive Average Pooling
* Lineer sınıflandırıcı

Ağ yapısı hızlı, hafif ve eğitim için uygun olacak şekilde tasarlanmıştır.

---

## 🔧 Eğitim (train.py)

Eğitim dosyası:

* Veri setini yükler
* Dönüşümleri (Resize, Normalize) uygular
* Modeli başlatır
* 5 epoch boyunca eğitir
* `model.pth` olarak kaydeder

Eğitimi başlatmak için:

```
python src/train.py
```

---

## 🎨 Uygulama Arayüzü (app.py)

PyQt6 ile hazırlanmış arayüz:

* Kullanıcı bir resim seçer
* Model resmi işler ve tahmin edilen sınıfı ekranda gösterir

Çalıştırmak için:

```
python src/app.py
```

---

## 🖼️ Veri Setini GitHub'a Yükleyemedim 

Veri seti büyük olduğu için GitHub repo limitsiz değil. Bunun yerine:
ekran görüntülerini ekledim . sorularınız için iletişime geçebilirsiniz

* `data/train/` klasör yapısının içine her sınıfı ayrı ayrı ekleyip çalıştırabilirsiniz 
* eğer projedeki veriseti size lazımsa kaggle.org dan indirebilirsiniz 

---

## 📌 Yeni Sınıf Eklemek

Yeni sınıf eklemek için:

1. `data/train/` içine yeni bir klasör oluştur (ör. `muz_saglikli`)
2. İçine resimleri koy
3. (Opsiyonel) `data/val/` içine aynı isimde bir klasör aç
4. `train.py` otomatik olarak sınıfı algılar

---

## 📦 Model Kaydetme

Eğitim sonunda model otomatik olarak kaydedilir:

```
src/model.pth
```



---

## 🛠 Gereken Kütüphaneler

```
pip install torch torchvision
pip install pyqt6
```

---

## 💡 Notlar

* `val` klasörüne resim koymak zorunlu değildir ama boş klasörler hata verebilir.
* Eğer val kullanmayacaksanız train.py içerisindeki val kodlarını silebilirsiniz.

---

## 📬 İletişim

Herhangi bir geliştirme önerisi veya hata bildirimi için issue açabilirsiniz.

---

**✔ Bu proje kişisel bir yapay zeka eğitim projesidir, isteyen herkes geliştirebilir.**
