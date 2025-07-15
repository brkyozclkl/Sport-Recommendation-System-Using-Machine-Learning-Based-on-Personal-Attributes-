# 🇹🇷 Spor Yetenek Tahmin Sistemi 🏃‍♂️

Türk insanlarının fiziksel özelliklerine ve geçmiş deneyimlerine dayalı olarak hangi spora daha yatkın olduğunu tahmin eden makine öğrenmesi sistemi.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost-orange.svg)](https://github.com/microsoft/LightGBM)

## 🎯 Özellikler

- 🇹🇷 **Türkiye'ye Özel**: Türk insanlarının fiziksel özellikleri ve spor kültürü
- 🤖 **5 Farklı ML Algoritması**: LightGBM, XGBoost, Random Forest, SVM, Neural Network
- 📊 **42 Farklı Özellik**: Demografik, fiziksel, performans, genetik, deneyim faktörleri
- 🏆 **10 Spor Kategorisi**: Futbol, basketbol, voleybol, güreş, yüzme ve daha fazlası
- 🎨 **İnteraktif Arayüz**: Streamlit ile modern web uygulaması
- 📈 **Görselleştirme**: Plotly ile detaylı analiz grafikleri
- 🎯 **%75 Doğruluk**: LightGBM ile yüksek performans

## 🎯 Proje Amacı

Bu proje, bir kişinin yaş, cinsiyet, boy, kilo, vücut tipi, genetik geçmişi ve daha önce yaptığı sporlar gibi bilgilerine dayanarak:
- Hangi spora daha yatkın olduğunu tahmin eder
- Spor önerilerini görselleştiren bir arayüz sunar
- Farklı makine öğrenmesi modellerini karşılaştırır

## 📁 Proje Yapısı

```
sporcu/
├── data/              # Veri setleri
├── models/            # Eğitilmiş modeller
├── notebooks/         # Jupyter notebook'lar
├── src/               # Kaynak kod
├── app/               # Streamlit uygulaması
├── requirements.txt   # Bağımlılıklar
└── README.md         # Proje dokümantasyonu
```

## 🚀 Kurulum

```bash
# Repository'yi klonla
git clone https://github.com/brkyozclkl/sporcu-yetenek-tahmin-sistemi.git
cd sporcu-yetenek-tahmin-sistemi

# Virtual environment oluştur (önerilen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt

# Veri setini oluştur (isteğe bağlı - örnek veri mevcut)
python src/data_generator.py

# Streamlit uygulamasını çalıştır
streamlit run app/main.py
```

## 🌐 Demo

Uygulamayı başlattıktan sonra `http://localhost:8501` adresinde test edebilirsiniz.

## 📊 Özellikler

- **Fiziksel Özellikler**: Yaş, cinsiyet, boy, kilo, BMI
- **Vücut Tipi**: Ektomorf, mezomorf, endomorf
- **Fiziksel Kabiliyetler**: Hız, kuvvet, dayanıklılık, esneklik
- **Genetik Faktörler**: Ailevi spor geçmişi
- **Deneyim**: Daha önce yapılan sporlar

## 🤖 Makine Öğrenmesi Modelleri

- Random Forest
- XGBoost
- LightGBM
- SVM
- Neural Network

## 🏆 Hedef Sporlar

- Koşu/Atletizm
- Futbol
- Basketbol
- Yüzme
- Voleybol
- Tenis
- Jimnastik
- Boks
- Güreş
- Bisiklet

## 👥 Kullanım Alanları

- Spor kulüpleri için yetenek keşfi
- Bireysel spor önerisi
- Akademik araştırma
- Kariyer gelişimi

## 🛠️ Geliştirme

Projeyi geliştirmek için:

1. Veri setini genişletin
2. Yeni özellikler ekleyin
3. Model performansını iyileştirin
4. Arayüzü zenginleştirin 