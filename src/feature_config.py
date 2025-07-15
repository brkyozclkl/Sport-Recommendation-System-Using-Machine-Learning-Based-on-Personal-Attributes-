"""
Spor Yetenek Tahmin Sistemi - Özellik Konfigürasyonu
Bu dosya, makine öğrenmesi modeli için kullanılacak tüm özellikleri tanımlar.
"""

# Temel demografik özellikler
DEMOGRAPHIC_FEATURES = {
    'yas': {
        'type': 'numeric',
        'range': (12, 50),
        'description': 'Yaş (yıl)',
        'importance': 'high'
    },
    'cinsiyet': {
        'type': 'categorical',
        'values': ['Erkek', 'Kadın'],
        'description': 'Cinsiyet',
        'importance': 'medium'
    },
    'boy': {
        'type': 'numeric',
        'range': (140, 220),
        'description': 'Boy (cm)',
        'importance': 'high'
    },
    'kilo': {
        'type': 'numeric',
        'range': (40, 150),
        'description': 'Kilo (kg)',
        'importance': 'high'
    },
    'bmi': {
        'type': 'calculated',
        'formula': 'kilo / (boy/100)^2',
        'description': 'Vücut Kitle İndeksi',
        'importance': 'high'
    }
}

# Vücut tipi ve fiziksel özellikler
PHYSICAL_FEATURES = {
    'vucut_tipi': {
        'type': 'categorical',
        'values': ['Ektomorf', 'Mezomorf', 'Endomorf'],
        'description': 'Vücut tipi (genetik yapı)',
        'importance': 'high'
    },
    'kas_orani': {
        'type': 'numeric',
        'range': (15, 45),
        'description': 'Kas oranı (%)',
        'importance': 'high'
    },
    'yag_orani': {
        'type': 'numeric',
        'range': (5, 35),
        'description': 'Yağ oranı (%)',
        'importance': 'medium'
    },
    'kemik_yogunlugu': {
        'type': 'categorical',
        'values': ['Düşük', 'Orta', 'Yüksek'],
        'description': 'Kemik yoğunluğu',
        'importance': 'medium'
    }
}

# Fiziksel kabiliyetler (0-10 skala)
PERFORMANCE_FEATURES = {
    'hiz': {
        'type': 'numeric',
        'range': (1, 10),
        'description': 'Hız kabiliyeti (1-10)',
        'importance': 'high'
    },
    'kuvvet': {
        'type': 'numeric',
        'range': (1, 10),
        'description': 'Kuvvet kabiliyeti (1-10)',
        'importance': 'high'
    },
    'dayaniklilik': {
        'type': 'numeric',
        'range': (1, 10),
        'description': 'Dayanıklılık kabiliyeti (1-10)',
        'importance': 'high'
    },
    'esneklik': {
        'type': 'numeric',
        'range': (1, 10),
        'description': 'Esneklik kabiliyeti (1-10)',
        'importance': 'medium'
    },
    'koordinasyon': {
        'type': 'numeric',
        'range': (1, 10),
        'description': 'Koordinasyon kabiliyeti (1-10)',
        'importance': 'high'
    },
    'denge': {
        'type': 'numeric',
        'range': (1, 10),
        'description': 'Denge kabiliyeti (1-10)',
        'importance': 'medium'
    },
    'reaksiyon_hizi': {
        'type': 'numeric',
        'range': (1, 10),
        'description': 'Reaksiyon hızı (1-10)',
        'importance': 'medium'
    }
}

# Genetik ve ailevi faktörler
GENETIC_FEATURES = {
    'ailevi_spor_gecmisi': {
        'type': 'categorical',
        'values': ['Yok', 'Var'],
        'description': 'Ailede profesyonel sporcu var mı?',
        'importance': 'medium'
    },
    'anne_spor_durumu': {
        'type': 'categorical',
        'values': ['Sedanter', 'Aktif', 'Sporcu'],
        'description': 'Annenin spor durumu',
        'importance': 'low'
    },
    'baba_spor_durumu': {
        'type': 'categorical',
        'values': ['Sedanter', 'Aktif', 'Sporcu'],
        'description': 'Babanın spor durumu',
        'importance': 'low'
    },
    'dominant_el': {
        'type': 'categorical',
        'values': ['Sağ', 'Sol', 'Ambidekstır'],
        'description': 'Dominant el',
        'importance': 'low'
    }
}

# Deneyim ve geçmiş sporlar
EXPERIENCE_FEATURES = {
    'spor_yili': {
        'type': 'numeric',
        'range': (0, 30),
        'description': 'Toplam spor deneyimi (yıl)',
        'importance': 'high'
    },
    'onceki_sporlar': {
        'type': 'multi_categorical',
        'values': ['Futbol', 'Basketbol', 'Voleybol', 'Tenis', 'Yüzme', 
                  'Atletizm', 'Jimnastik', 'Boks', 'Güreş', 'Bisiklet', 'Hiçbiri'],
        'description': 'Daha önce yapılan sporlar',
        'importance': 'medium'
    },
    'en_basarili_spor': {
        'type': 'categorical',
        'values': ['Futbol', 'Basketbol', 'Voleybol', 'Tenis', 'Yüzme', 
                  'Atletizm', 'Jimnastik', 'Boks', 'Güreş', 'Bisiklet', 'Hiçbiri'],
        'description': 'En başarılı olunan spor',
        'importance': 'high'
    },
    'yaralanma_gecmisi': {
        'type': 'categorical',
        'values': ['Yok', 'Hafif', 'Orta', 'Ağır'],
        'description': 'Spor yaralanması geçmişi',
        'importance': 'medium'
    }
}

# Psikolojik ve mental faktörler
PSYCHOLOGICAL_FEATURES = {
    'stres_toleransi': {
        'type': 'numeric',
        'range': (1, 10),
        'description': 'Stres toleransı (1-10)',
        'importance': 'medium'
    },
    'takım_oyunu_tercihi': {
        'type': 'categorical',
        'values': ['Bireysel', 'Takım', 'Karma'],
        'description': 'Takım oyunu tercihi',
        'importance': 'high'
    },
    'yarışma_tutkusu': {
        'type': 'numeric',
        'range': (1, 10),
        'description': 'Yarışma tutkusu (1-10)',
        'importance': 'medium'
    },
    'konsantrasyon': {
        'type': 'numeric',
        'range': (1, 10),
        'description': 'Konsantrasyon yeteneği (1-10)',
        'importance': 'medium'
    }
}

# Çevresel faktörler
ENVIRONMENTAL_FEATURES = {
    'coğrafi_konum': {
        'type': 'categorical',
        'values': ['Marmara', 'Ege', 'Akdeniz', 'İç Anadolu', 'Karadeniz', 'Doğu Anadolu', 'Güneydoğu Anadolu'],
        'description': 'Türkiye coğrafi bölgesi',
        'importance': 'low'
    },
    'ekonomik_durum': {
        'type': 'categorical',
        'values': ['Düşük', 'Orta', 'Yüksek'],
        'description': 'Ekonomik durum',
        'importance': 'low'
    },
    'tesis_erisimi': {
        'type': 'categorical',
        'values': ['Zor', 'Orta', 'Kolay'],
        'description': 'Spor tesislerine erişim',
        'importance': 'low'
    }
}

# Hedef sporlar - Türkiye'deki popülerlik sırasına göre
TARGET_SPORTS = {
    'Futbol': {
        'key_features': ['hiz', 'dayaniklilik', 'koordinasyon', 'takım_oyunu_tercihi'],
        'preferred_body_type': ['Mezomorf'],
        'description': 'Türkiye\'nin en popüler sporu - Futbol',
        'popularity_weight': 0.30  # Türkiye'deki popülerlik ağırlığı
    },
    'Koşu/Atletizm': {
        'key_features': ['hiz', 'dayaniklilik', 'boy', 'kilo'],
        'preferred_body_type': ['Ektomorf'],
        'description': 'Koşu ve atletizm sporları',
        'popularity_weight': 0.15
    },
    'Basketbol': {
        'key_features': ['boy', 'hiz', 'koordinasyon', 'takım_oyunu_tercihi'],
        'preferred_body_type': ['Ektomorf', 'Mezomorf'],
        'description': 'Türkiye\'de ikinci popüler takım sporu - Basketbol',
        'popularity_weight': 0.12
    },
    'Voleybol': {
        'key_features': ['boy', 'hiz', 'koordinasyon', 'takım_oyunu_tercihi'],
        'preferred_body_type': ['Ektomorf'],
        'description': 'Türkiye\'de özellikle kadınlar arasında popüler - Voleybol',
        'popularity_weight': 0.10
    },
    'Yüzme': {
        'key_features': ['boy', 'dayaniklilik', 'kuvvet', 'esneklik'],
        'preferred_body_type': ['Ektomorf', 'Mezomorf'],
        'description': 'Kıyı bölgelerde popüler - Yüzme',
        'popularity_weight': 0.08
    },
    'Güreş': {
        'key_features': ['kuvvet', 'dayaniklilik', 'denge', 'esneklik'],
        'preferred_body_type': ['Mezomorf', 'Endomorf'],
        'description': 'Türkiye\'nin geleneksel ve milli sporu - Güreş',
        'popularity_weight': 0.08
    },
    'Tenis': {
        'key_features': ['hiz', 'koordinasyon', 'esneklik', 'reaksiyon_hizi'],
        'preferred_body_type': ['Mezomorf'],
        'description': 'Üst gelir gruplarında popüler - Tenis',
        'popularity_weight': 0.06
    },
    'Bisiklet': {
        'key_features': ['dayaniklilik', 'kuvvet', 'denge'],
        'preferred_body_type': ['Ektomorf', 'Mezomorf'],
        'description': 'Rekreasyonel olarak popüler - Bisiklet',
        'popularity_weight': 0.05
    },
    'Jimnastik': {
        'key_features': ['esneklik', 'koordinasyon', 'denge', 'kuvvet'],
        'preferred_body_type': ['Ektomorf'],
        'description': 'Genç yaşlarda popüler - Jimnastik',
        'popularity_weight': 0.04
    },
    'Boks': {
        'key_features': ['kuvvet', 'hiz', 'reaksiyon_hizi', 'dayaniklilik'],
        'preferred_body_type': ['Mezomorf'],
        'description': 'Geleneksel dövüş sporu - Boks',
        'popularity_weight': 0.03
    }
}

# Tüm özellikleri birleştir
ALL_FEATURES = {
    **DEMOGRAPHIC_FEATURES,
    **PHYSICAL_FEATURES,
    **PERFORMANCE_FEATURES,
    **GENETIC_FEATURES,
    **EXPERIENCE_FEATURES,
    **PSYCHOLOGICAL_FEATURES,
    **ENVIRONMENTAL_FEATURES
}

# Özellik grupları
FEATURE_GROUPS = {
    'demografik': DEMOGRAPHIC_FEATURES,
    'fiziksel': PHYSICAL_FEATURES,
    'performans': PERFORMANCE_FEATURES,
    'genetik': GENETIC_FEATURES,
    'deneyim': EXPERIENCE_FEATURES,
    'psikolojik': PSYCHOLOGICAL_FEATURES,
    'çevresel': ENVIRONMENTAL_FEATURES
}

# Özellik öncelik sıralaması
FEATURE_IMPORTANCE_ORDER = {
    'high': ['yas', 'boy', 'kilo', 'bmi', 'vucut_tipi', 'kas_orani', 'hiz', 
             'kuvvet', 'dayaniklilik', 'koordinasyon', 'spor_yili', 'en_basarili_spor',
             'takım_oyunu_tercihi'],
    'medium': ['cinsiyet', 'yag_orani', 'kemik_yogunlugu', 'esneklik', 'denge', 
               'reaksiyon_hizi', 'ailevi_spor_gecmisi', 'onceki_sporlar', 
               'yaralanma_gecmisi', 'stres_toleransi', 'yarışma_tutkusu', 'konsantrasyon'],
    'low': ['anne_spor_durumu', 'baba_spor_durumu', 'dominant_el', 'coğrafi_konum',
            'ekonomik_durum', 'tesis_erisimi']
} 