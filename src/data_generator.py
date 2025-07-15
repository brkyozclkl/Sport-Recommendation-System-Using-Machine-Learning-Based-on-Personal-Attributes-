"""
Sentetik Veri Üretimi - Spor Yetenek Tahmin Sistemi
Bu dosya, makine öğrenmesi modeli için gerçekçi sentetik veri üretir.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from typing import Dict, List, Tuple
import json

try:
    from feature_config import (
        ALL_FEATURES, TARGET_SPORTS, FEATURE_GROUPS,
        DEMOGRAPHIC_FEATURES, PHYSICAL_FEATURES, PERFORMANCE_FEATURES,
        GENETIC_FEATURES, EXPERIENCE_FEATURES, PSYCHOLOGICAL_FEATURES,
        ENVIRONMENTAL_FEATURES
    )
except ImportError:
    from src.feature_config import (
        ALL_FEATURES, TARGET_SPORTS, FEATURE_GROUPS,
        DEMOGRAPHIC_FEATURES, PHYSICAL_FEATURES, PERFORMANCE_FEATURES,
        GENETIC_FEATURES, EXPERIENCE_FEATURES, PSYCHOLOGICAL_FEATURES,
        ENVIRONMENTAL_FEATURES
    )

class SportsDataGenerator:
    """Spor yetenek tahmin sistemi için sentetik veri üretici"""
    
    def __init__(self, seed: int = 42):
        """
        Veri üretici sınıfını başlatır
        
        Args:
            seed: Rastgele sayı üreteci için seed değeri
        """
        self.fake = Faker('tr_TR')  # Türkçe locale
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        
    def _generate_demographic_features(self) -> Dict:
        """Demografik özellikler üretir - Türk insanlarının özelliklerine göre"""
        yas = random.randint(12, 50)
        cinsiyet = random.choice(['Erkek', 'Kadın'])
        
        # Türk insanlarının ortalama boy-kilo değerleri
        if cinsiyet == 'Erkek':
            # Türk erkeklerinin ortalama boyu: 173.7 cm
            boy = np.random.normal(173.7, 7.5)
            kilo_base = (boy - 100) * 0.85  # Türk erkekleri için ayarlandı
        else:
            # Türk kadınlarının ortalama boyu: 161.4 cm
            boy = np.random.normal(161.4, 6.5)
            kilo_base = (boy - 100) * 0.8  # Türk kadınları için ayarlandı
            
        # Yaş faktörü
        if yas < 18:
            kilo_base *= 0.85
        elif yas > 35:
            kilo_base *= 1.1
            
        boy = max(140, min(220, boy))
        kilo = max(40, min(150, kilo_base + random.uniform(-10, 10)))
        bmi = kilo / ((boy / 100) ** 2)
        
        return {
            'yas': round(yas),
            'cinsiyet': cinsiyet,
            'boy': round(boy),
            'kilo': round(kilo, 1),
            'bmi': round(bmi, 1)
        }
    
    def _generate_physical_features(self, demographics: Dict) -> Dict:
        """Fiziksel özellikler üretir"""
        bmi = demographics['bmi']
        yas = demographics['yas']
        cinsiyet = demographics['cinsiyet']
        
        # BMI'ye göre vücut tipi dağılımı - Türk toplumuna göre
        if bmi < 20:
            # Türk toplumunda ektomorf oranı daha düşük
            vucut_tipi = np.random.choice(['Ektomorf', 'Mezomorf'], p=[0.7, 0.3])
            kas_orani = random.uniform(15, 25)
            yag_orani = random.uniform(5, 15)
        elif bmi < 25:
            # Orta BMI'de mezomorf daha yaygın
            vucut_tipi = np.random.choice(['Ektomorf', 'Mezomorf', 'Endomorf'], p=[0.2, 0.6, 0.2])
            kas_orani = random.uniform(20, 35)
            yag_orani = random.uniform(10, 20)
        else:
            # Yüksek BMI'de endomorf daha yaygın (Türk toplumunda obezite artışı)
            vucut_tipi = np.random.choice(['Mezomorf', 'Endomorf'], p=[0.4, 0.6])
            kas_orani = random.uniform(25, 40)
            yag_orani = random.uniform(15, 30)
            
        # Cinsiyet faktörü
        if cinsiyet == 'Kadın':
            kas_orani *= 0.8
            yag_orani *= 1.2
            
        # Yaş faktörü
        if yas > 30:
            kas_orani *= 0.95
            yag_orani *= 1.1
        
        kemik_yogunlugu = random.choice(['Düşük', 'Orta', 'Yüksek'])
        
        return {
            'vucut_tipi': vucut_tipi,
            'kas_orani': round(kas_orani, 1),
            'yag_orani': round(yag_orani, 1),
            'kemik_yogunlugu': kemik_yogunlugu
        }
    
    def _generate_performance_features(self, demographics: Dict, physical: Dict) -> Dict:
        """Performans özelliklerini üretir"""
        yas = demographics['yas']
        cinsiyet = demographics['cinsiyet']
        vucut_tipi = physical['vucut_tipi']
        kas_orani = physical['kas_orani']
        
        # Vücut tipine göre temel yetenekler
        if vucut_tipi == 'Ektomorf':
            hiz_base = 7
            kuvvet_base = 5
            dayaniklilik_base = 8
            esneklik_base = 7
        elif vucut_tipi == 'Mezomorf':
            hiz_base = 6
            kuvvet_base = 8
            dayaniklilik_base = 6
            esneklik_base = 6
        else:  # Endomorf
            hiz_base = 4
            kuvvet_base = 7
            dayaniklilik_base = 5
            esneklik_base = 5
            
        # Yaş faktörü
        if yas < 18:
            multiplier = 0.8
        elif yas < 25:
            multiplier = 1.0
        elif yas < 35:
            multiplier = 0.9
        else:
            multiplier = 0.75
            
        # Cinsiyet faktörü
        if cinsiyet == 'Kadın':
            kuvvet_base *= 0.8
            esneklik_base *= 1.2
            
        # Kas oranı etkisi
        kas_multiplier = kas_orani / 25
        
        features = {
            'hiz': max(1, min(10, hiz_base * multiplier + random.uniform(-1, 1))),
            'kuvvet': max(1, min(10, kuvvet_base * multiplier * kas_multiplier + random.uniform(-1, 1))),
            'dayaniklilik': max(1, min(10, dayaniklilik_base * multiplier + random.uniform(-1, 1))),
            'esneklik': max(1, min(10, esneklik_base * multiplier + random.uniform(-1, 1))),
            'koordinasyon': max(1, min(10, random.uniform(4, 8) * multiplier)),
            'denge': max(1, min(10, random.uniform(4, 8) * multiplier)),
            'reaksiyon_hizi': max(1, min(10, random.uniform(4, 8) * multiplier))
        }
        
        return {k: round(v, 1) for k, v in features.items()}
    
    def _generate_genetic_features(self) -> Dict:
        """Genetik özellikler üretir - Türk toplumuna göre"""
        # Türkiye'de ailevi spor geçmişi oranı (daha düşük)
        ailevi_spor_gecmisi = np.random.choice(['Yok', 'Var'], p=[0.75, 0.25])
        
        # Türk toplumunda kadın-erkek spor katılımı farklılıkları
        if ailevi_spor_gecmisi == 'Var':
            # Geleneksel olarak erkek spor katılımı daha yüksek
            anne_spor = np.random.choice(['Sedanter', 'Aktif', 'Sporcu'], p=[0.4, 0.45, 0.15])
            baba_spor = np.random.choice(['Sedanter', 'Aktif', 'Sporcu'], p=[0.25, 0.55, 0.20])
        else:
            # Spor geçmişi olmayan ailelerde daha düşük aktivite
            anne_spor = np.random.choice(['Sedanter', 'Aktif', 'Sporcu'], p=[0.65, 0.30, 0.05])
            baba_spor = np.random.choice(['Sedanter', 'Aktif', 'Sporcu'], p=[0.50, 0.40, 0.10])
            
        # Türk toplumunda solakların oranı: ~%10
        dominant_el = np.random.choice(['Sağ', 'Sol'], p=[0.90, 0.10])
        
        return {
            'ailevi_spor_gecmisi': ailevi_spor_gecmisi,
            'anne_spor_durumu': anne_spor,
            'baba_spor_durumu': baba_spor,
            'dominant_el': dominant_el
        }
    
    def _generate_experience_features(self, demographics: Dict) -> Dict:
        """Deneyim özelliklerini üretir"""
        yas = demographics['yas']
        
        # Yaşa göre spor deneyimi
        max_spor_yili = max(0, yas - 10)
        spor_yili = random.randint(0, max_spor_yili)
        
        # Spor geçmişi - Türkiye'deki popüler sporlar
        if spor_yili == 0:
            onceki_sporlar = ['Hiçbiri']
            en_basarili_spor = 'Hiçbiri'
            yaralanma_gecmisi = 'Yok'
        else:
            num_sports = min(random.randint(1, 4), spor_yili)
            # Türkiye'deki popüler sporlar ve yaygınlık oranları
            turkiye_sporlari = ['Futbol', 'Basketbol', 'Voleybol', 'Yüzme', 
                              'Atletizm', 'Tenis', 'Güreş', 'Bisiklet', 'Jimnastik', 'Boks']
            spor_agirliklari = [0.30, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
            
            # Popülerlik oranına göre spor seçimi
            onceki_sporlar = np.random.choice(turkiye_sporlari, 
                                            size=num_sports, 
                                            replace=False, 
                                            p=spor_agirliklari).tolist()
            en_basarili_spor = random.choice(onceki_sporlar)
            
            # Yaralanma riski spor yılı ile artar
            if spor_yili > 10:
                yaralanma_gecmisi = random.choice(['Yok', 'Hafif', 'Orta', 'Ağır'])
            elif spor_yili > 5:
                yaralanma_gecmisi = random.choice(['Yok', 'Hafif', 'Orta'])
            else:
                yaralanma_gecmisi = random.choice(['Yok', 'Hafif'])
        
        return {
            'spor_yili': spor_yili,
            'onceki_sporlar': onceki_sporlar,
            'en_basarili_spor': en_basarili_spor,
            'yaralanma_gecmisi': yaralanma_gecmisi
        }
    
    def _generate_psychological_features(self, demographics: Dict, performance: Dict) -> Dict:
        """Psikolojik özellikler üretir"""
        yas = demographics['yas']
        
        # Yaş ile stres toleransı artar
        stres_base = min(8, 4 + (yas - 12) * 0.1)
        stres_toleransi = max(1, min(10, stres_base + random.uniform(-2, 2)))
        
        # Takım oyunu tercihi
        takım_oyunu_tercihi = random.choice(['Bireysel', 'Takım', 'Karma'])
        
        # Yarışma tutkusu performansla ilişkili
        avg_performance = (performance['hiz'] + performance['kuvvet'] + performance['dayaniklilik']) / 3
        yarışma_base = avg_performance * 0.8
        yarışma_tutkusu = max(1, min(10, yarışma_base + random.uniform(-2, 2)))
        
        # Konsantrasyon
        konsantrasyon = max(1, min(10, random.uniform(4, 8)))
        
        return {
            'stres_toleransi': round(stres_toleransi, 1),
            'takım_oyunu_tercihi': takım_oyunu_tercihi,
            'yarışma_tutkusu': round(yarışma_tutkusu, 1),
            'konsantrasyon': round(konsantrasyon, 1)
        }
    
    def _generate_environmental_features(self) -> Dict:
        """Çevresel özellikler üretir - Türkiye coğrafyasına göre"""
        # Türkiye'nin coğrafi bölgeleri ve nüfus yoğunluğuna göre ağırlıklandırma
        cografi_secenekler = [
            'Marmara', 'Ege', 'Akdeniz', 'İç Anadolu', 'Karadeniz', 
            'Doğu Anadolu', 'Güneydoğu Anadolu'
        ]
        # Nüfus yoğunluğuna göre ağırlıklandırma
        cografi_agirlik = [0.25, 0.15, 0.12, 0.18, 0.12, 0.08, 0.10]
        
        # Türkiye'nin sosyoekonomik durumu
        ekonomik_secenekler = ['Düşük', 'Orta', 'Yüksek']
        ekonomik_agirlik = [0.35, 0.50, 0.15]  # Türkiye'nin gelir dağılımı
        
        # Spor tesisi erişimi (şehir büyüklüğüne göre)
        tesis_secenekler = ['Zor', 'Orta', 'Kolay']
        tesis_agirlik = [0.30, 0.45, 0.25]  # Türkiye'deki spor tesisi dağılımı
        
        return {
            'coğrafi_konum': np.random.choice(cografi_secenekler, p=cografi_agirlik),
            'ekonomik_durum': np.random.choice(ekonomik_secenekler, p=ekonomik_agirlik),
            'tesis_erisimi': np.random.choice(tesis_secenekler, p=tesis_agirlik)
        }
    
    def _calculate_sport_compatibility(self, person_data: Dict) -> Dict:
        """Kişinin hangi spora ne kadar uygun olduğunu hesaplar"""
        compatibilities = {}
        
        for sport, sport_info in TARGET_SPORTS.items():
            score = 0
            max_score = 0
            
            # Anahtar özellikler skoru
            for feature in sport_info['key_features']:
                if feature in person_data:
                    if feature in ['boy', 'kilo']:
                        # Boy-kilo için normalize et
                        if feature == 'boy':
                            normalized = (person_data[feature] - 160) / 30  # 160-190 arası normalize
                        else:
                            normalized = (80 - person_data[feature]) / 30  # Düşük kilo avantajlı
                        score += max(0, min(10, 5 + normalized * 2))
                    elif feature == 'takım_oyunu_tercihi':
                        # Takım sporları için bonus
                        team_sports = ['Futbol', 'Basketbol', 'Voleybol']
                        if sport in team_sports and person_data[feature] in ['Takım', 'Karma']:
                            score += 8
                        elif sport not in team_sports and person_data[feature] in ['Bireysel', 'Karma']:
                            score += 8
                        else:
                            score += 4
                    else:
                        score += person_data[feature]
                    max_score += 10
            
            # Vücut tipi uygunluğu
            if person_data['vucut_tipi'] in sport_info['preferred_body_type']:
                score += 15
            else:
                score += 5
            max_score += 15
            
            # Normalize et (0-100)
            compatibility = (score / max_score) * 100 if max_score > 0 else 0
            compatibilities[sport] = round(compatibility, 1)
        
        return compatibilities
    
    def generate_single_person(self) -> Dict:
        """Tek bir kişi için veri üretir"""
        # Demografik özellikler
        demographics = self._generate_demographic_features()
        
        # Fiziksel özellikler
        physical = self._generate_physical_features(demographics)
        
        # Performans özellikleri
        performance = self._generate_performance_features(demographics, physical)
        
        # Genetik özellikler
        genetic = self._generate_genetic_features()
        
        # Deneyim özellikleri
        experience = self._generate_experience_features(demographics)
        
        # Psikolojik özellikler
        psychological = self._generate_psychological_features(demographics, performance)
        
        # Çevresel özellikler
        environmental = self._generate_environmental_features()
        
        # Tüm özellikleri birleştir
        person_data = {
            **demographics,
            **physical,
            **performance,
            **genetic,
            **experience,
            **psychological,
            **environmental
        }
        
        # Spor uygunluk skorlarını hesapla
        sport_scores = self._calculate_sport_compatibility(person_data)
        
        # En uygun sporu belirle
        best_sport = max(sport_scores.keys(), key=lambda x: sport_scores[x])
        person_data['tavsiye_edilen_spor'] = best_sport
        person_data['spor_skorlari'] = sport_scores
        
        return person_data
    
    def generate_dataset(self, num_people: int = 100) -> pd.DataFrame:
        """Belirtilen sayıda kişi için veri seti üretir"""
        data = []
        
        for i in range(num_people):
            person = self.generate_single_person()
            # Spor skorlarını ayrı sütunlara çevir
            sport_scores = person.pop('spor_skorlari')
            for sport, score in sport_scores.items():
                person[f'skor_{sport.replace("/", "_").replace(" ", "_").lower()}'] = score
            
            data.append(person)
        
        return pd.DataFrame(data)
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """Veri setini dosyaya kaydeder"""
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Veri seti kaydedildi: {filename}")
        print(f"Veri boyutu: {df.shape}")
        print(f"Özellik sayısı: {len(df.columns)}")

# Kullanım örneği
if __name__ == "__main__":
    generator = SportsDataGenerator(seed=42)
    
    # 200 kişilik veri seti oluştur
    dataset = generator.generate_dataset(200)
    
    # Veri setini kaydet
    generator.save_dataset(dataset, "data/sporcu_dataset.csv")
    
    # Temel istatistikler
    print("\n=== Veri Seti Özeti ===")
    print(f"Toplam kişi sayısı: {len(dataset)}")
    print(f"Ortalama yaş: {dataset['yas'].mean():.1f}")
    print(f"Cinsiyet dağılımı:")
    print(dataset['cinsiyet'].value_counts())
    print(f"\nVücut tipi dağılımı:")
    print(dataset['vucut_tipi'].value_counts())
    print(f"\nEn çok tavsiye edilen sporlar:")
    print(dataset['tavsiye_edilen_spor'].value_counts()) 