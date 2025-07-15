"""
Büyük Veri Seti Üretimi - Spor Yetenek Tahmin Sistemi
Bu dosya, büyük veri setlerini parçalar halinde üretir.
"""

import pandas as pd
import numpy as np
from data_generator import SportsDataGenerator
import os
from typing import List
import time

class BulkDataGenerator:
    """Büyük veri setlerini parçalar halinde üretir"""
    
    def __init__(self, batch_size: int = 50, seed: int = 42):
        """
        Bulk veri üretici sınıfını başlatır
        
        Args:
            batch_size: Her seferinde üretilecek kişi sayısı
            seed: Rastgele sayı üreteci için seed değeri
        """
        self.batch_size = batch_size
        self.base_seed = seed
        self.generator = SportsDataGenerator(seed=seed)
        
    def generate_batch(self, batch_num: int) -> pd.DataFrame:
        """Tek bir batch veri üretir"""
        # Her batch için farklı seed kullan
        batch_seed = self.base_seed + batch_num * 1000
        self.generator = SportsDataGenerator(seed=batch_seed)
        
        return self.generator.generate_dataset(self.batch_size)
    
    def generate_large_dataset(self, total_size: int, 
                             save_path: str = "data/sporcu_dataset_large.csv",
                             progress_callback=None) -> pd.DataFrame:
        """
        Büyük veri setini parçalar halinde üretir
        
        Args:
            total_size: Toplam kişi sayısı
            save_path: Kaydedilecek dosya yolu
            progress_callback: İlerleme callback fonksiyonu
        """
        print(f"🚀 {total_size} kişilik veri seti üretiliyor...")
        print(f"📦 Batch boyutu: {self.batch_size}")
        
        all_data = []
        batches = (total_size + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(batches):
            start_time = time.time()
            
            # Son batch için kalan kişi sayısını hesapla
            remaining = total_size - (batch_num * self.batch_size)
            current_batch_size = min(self.batch_size, remaining)
            
            # Batch üret
            if current_batch_size > 0:
                # Bu batch için özel generator
                batch_generator = SportsDataGenerator(seed=self.base_seed + batch_num * 1000)
                batch_data = batch_generator.generate_dataset(current_batch_size)
                all_data.append(batch_data)
            
            # İlerleme bilgisi
            completed = (batch_num + 1) * self.batch_size
            if completed > total_size:
                completed = total_size
                
            elapsed_time = time.time() - start_time
            
            print(f"✅ Batch {batch_num + 1}/{batches} tamamlandı: "
                  f"{completed}/{total_size} kişi ({elapsed_time:.2f}s)")
            
            if progress_callback:
                progress_callback(completed, total_size, batch_num + 1, batches)
        
        # Tüm batch'leri birleştir
        print("🔄 Batch'ler birleştiriliyor...")
        final_dataset = pd.concat(all_data, ignore_index=True)
        
        # Dosyaya kaydet
        print(f"💾 Veri seti kaydediliyor: {save_path}")
        final_dataset.to_csv(save_path, index=False, encoding='utf-8')
        
        # Özet bilgiler
        print("\n" + "="*50)
        print("📊 BÜYÜK VERİ SETİ ÖZET BİLGİLERİ")
        print("="*50)
        print(f"📈 Toplam kişi sayısı: {len(final_dataset)}")
        print(f"📊 Özellik sayısı: {len(final_dataset.columns)}")
        print(f"📁 Dosya boyutu: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
        print(f"📋 Ortalama yaş: {final_dataset['yas'].mean():.1f}")
        
        print(f"\n🏆 En çok tavsiye edilen sporlar (Türkiye'ye özel):")
        sport_counts = final_dataset['tavsiye_edilen_spor'].value_counts()
        for i, (sport, count) in enumerate(sport_counts.head(5).items(), 1):
            print(f"   {i}. {sport}: {count} kişi (%{count/len(final_dataset)*100:.1f})")
        
        print(f"\n👥 Cinsiyet dağılımı:")
        gender_counts = final_dataset['cinsiyet'].value_counts()
        for gender, count in gender_counts.items():
            print(f"   {gender}: {count} kişi (%{count/len(final_dataset)*100:.1f})")
        
        return final_dataset
    
    def generate_progressive_dataset(self, size_list: List[int],
                                   base_path: str = "data/sporcu_dataset"):
        """
        Farklı boyutlarda veri setleri üretir
        
        Args:
            size_list: Üretilecek boyutlar listesi [100, 200, 500]
            base_path: Temel dosya yolu
        """
        for size in size_list:
            print(f"\n🎯 {size} kişilik veri seti üretiliyor...")
            
            dataset = self.generate_large_dataset(
                total_size=size,
                save_path=f"{base_path}_{size}.csv"
            )
            
            print(f"✅ {size} kişilik veri seti tamamlandı!\n")

# Kullanım fonksiyonları
def generate_500_dataset():
    """500 kişilik veri seti üretir"""
    bulk_generator = BulkDataGenerator(batch_size=50, seed=42)
    return bulk_generator.generate_large_dataset(
        total_size=500,
        save_path="data/sporcu_dataset_500.csv"
    )

def generate_multiple_datasets():
    """Farklı boyutlarda veri setleri üretir"""
    bulk_generator = BulkDataGenerator(batch_size=50, seed=42)
    bulk_generator.generate_progressive_dataset(
        size_list=[200, 300, 500],
        base_path="data/sporcu_dataset"
    )

# Kullanım örneği
if __name__ == "__main__":
    print("🇹🇷 Spor Yetenek Tahmin Sistemi - Türkiye'ye Özel Büyük Veri Seti Üretimi")
    print("="*70)
    
    # 500 kişilik Türkiye'ye özel veri seti üret
    generate_500_dataset()
    
    print("\n🎉 Türkiye'ye özel veri seti üretimi tamamlandı!")
    print("📁 Dosya: data/sporcu_dataset_500.csv")
    print("🇹🇷 Özellikler: Türk insanlarının fiziksel özellikleri, popüler sporlar, coğrafi bölgeler")
    print("🌐 Streamlit uygulamasını yeniden başlatın") 