"""
Veri Görselleştirme - Spor Yetenek Tahmin Sistemi
Bu dosya, veri setini görselleştiren ve analiz eden araçları içerir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

class SportsDataVisualizer:
    """Spor yetenek tahmin sistemi veri görselleştirici sınıfı"""
    
    def __init__(self, data_path: str = "data/sporcu_dataset_500.csv"):
        """
        Veri görselleştirici sınıfını başlatır
        
        Args:
            data_path: Veri seti dosya yolu
        """
        self.data_path = data_path
        self.data = None
        self.sport_colors = {
            'Koşu/Atletizm': '#FF6B6B',
            'Futbol': '#4ECDC4',
            'Basketbol': '#45B7D1',
            'Yüzme': '#96CEB4',
            'Voleybol': '#FFEAA7',
            'Tenis': '#DDA0DD',
            'Jimnastik': '#FFB6C1',
            'Boks': '#F4A460',
            'Güreş': '#8B4513',
            'Bisiklet': '#20B2AA'
        }
        
    def load_data(self):
        """Veri setini yükler"""
        print("Veri seti yükleniyor...")
        self.data = pd.read_csv(self.data_path)
        print(f"Veri boyutu: {self.data.shape}")
        return self.data
    
    def create_demographic_analysis(self):
        """Demografik analiz grafikleri oluşturur"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Demografik Analiz', fontsize=16, fontweight='bold')
        
        # Yaş dağılımı
        axes[0, 0].hist(self.data['yas'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Yaş Dağılımı')
        axes[0, 0].set_xlabel('Yaş')
        axes[0, 0].set_ylabel('Kişi Sayısı')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cinsiyet dağılımı
        gender_counts = self.data['cinsiyet'].value_counts()
        axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                       colors=['lightcoral', 'lightblue'])
        axes[0, 1].set_title('Cinsiyet Dağılımı')
        
        # BMI dağılımı
        axes[1, 0].hist(self.data['bmi'], bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('BMI Dağılımı')
        axes[1, 0].set_xlabel('BMI')
        axes[1, 0].set_ylabel('Kişi Sayısı')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Vücut tipi dağılımı
        body_type_counts = self.data['vucut_tipi'].value_counts()
        axes[1, 1].bar(body_type_counts.index, body_type_counts.values, 
                       color=['orange', 'purple', 'green'])
        axes[1, 1].set_title('Vücut Tipi Dağılımı')
        axes[1, 1].set_xlabel('Vücut Tipi')
        axes[1, 1].set_ylabel('Kişi Sayısı')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_sport_distribution_analysis(self):
        """Spor dağılımı analizi"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Spor Dağılımı Analizi', fontsize=16, fontweight='bold')
        
        # Tavsiye edilen spor dağılımı
        sport_counts = self.data['tavsiye_edilen_spor'].value_counts()
        colors = [self.sport_colors.get(sport, '#888888') for sport in sport_counts.index]
        
        axes[0].barh(sport_counts.index, sport_counts.values, color=colors)
        axes[0].set_title('Tavsiye Edilen Spor Dağılımı')
        axes[0].set_xlabel('Kişi Sayısı')
        axes[0].grid(True, alpha=0.3)
        
        # Spor skorları kutu grafikleri
        score_columns = [col for col in self.data.columns if col.startswith('skor_')]
        score_data = []
        sport_names = []
        
        for col in score_columns:
            sport_name = col.replace('skor_', '').replace('_', ' ').title()
            sport_names.append(sport_name)
            score_data.append(self.data[col])
        
        axes[1].boxplot(score_data, labels=sport_names)
        axes[1].set_title('Spor Skorları Dağılımı')
        axes[1].set_ylabel('Skor')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/sport_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_performance_analysis(self):
        """Performans analizi grafikleri"""
        performance_features = ['hiz', 'kuvvet', 'dayaniklilik', 'esneklik', 
                               'koordinasyon', 'denge', 'reaksiyon_hizi']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performans Analizi', fontsize=16, fontweight='bold')
        
        # Performans özelliklerinin korelasyon matrisi
        perf_corr = self.data[performance_features].corr()
        sns.heatmap(perf_corr, annot=True, cmap='coolwarm', center=0,
                   ax=axes[0, 0], fmt='.2f')
        axes[0, 0].set_title('Performans Özellikleri Korelasyonu')
        
        # Vücut tipine göre performans
        body_type_perf = self.data.groupby('vucut_tipi')[performance_features].mean()
        body_type_perf.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Vücut Tipine Göre Ortalama Performans')
        axes[0, 1].set_xlabel('Vücut Tipi')
        axes[0, 1].set_ylabel('Ortalama Skor')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cinsiyete göre performans
        gender_perf = self.data.groupby('cinsiyet')[performance_features].mean()
        gender_perf.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Cinsiyete Göre Ortalama Performans')
        axes[1, 0].set_xlabel('Cinsiyet')
        axes[1, 0].set_ylabel('Ortalama Skor')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Yaş gruplarına göre performans
        self.data['yas_grubu'] = pd.cut(self.data['yas'], 
                                       bins=[0, 18, 25, 35, 50], 
                                       labels=['12-18', '19-25', '26-35', '36-50'])
        age_perf = self.data.groupby('yas_grubu')[performance_features].mean()
        age_perf.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Yaş Grubuna Göre Ortalama Performans')
        axes[1, 1].set_xlabel('Yaş Grubu')
        axes[1, 1].set_ylabel('Ortalama Skor')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_scatter_plot(self):
        """Etkileşimli scatter plot oluşturur"""
        # Boy-kilo-BMI scatter plot
        fig = px.scatter(self.data, 
                        x='boy', 
                        y='kilo',
                        color='tavsiye_edilen_spor',
                        size='bmi',
                        hover_data=['yas', 'cinsiyet', 'vucut_tipi'],
                        title='Boy-Kilo-BMI İlişkisi (Tavsiye Edilen Spora Göre)',
                        color_discrete_map=self.sport_colors,
                        width=1000, height=600)
        
        fig.update_layout(
            xaxis_title='Boy (cm)',
            yaxis_title='Kilo (kg)',
            font=dict(size=12)
        )
        
        fig.write_html('visualizations/interactive_scatter.html')
        fig.show()
        
    def create_radar_chart(self):
        """Spor türlerine göre radar chart oluşturur"""
        performance_features = ['hiz', 'kuvvet', 'dayaniklilik', 'esneklik', 
                               'koordinasyon', 'denge', 'reaksiyon_hizi']
        
        sport_performance = self.data.groupby('tavsiye_edilen_spor')[performance_features].mean()
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=list(sport_performance.index[:6]),
            specs=[[{"type": "polar"}] * 3] * 2
        )
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, (sport, row) in enumerate(sport_performance.head(6).iterrows()):
            row_idx = i // 3 + 1
            col_idx = i % 3 + 1
            
            fig.add_trace(go.Scatterpolar(
                r=row.values,
                theta=performance_features,
                fill='toself',
                name=sport,
                line_color=colors[i]
            ), row=row_idx, col=col_idx)
        
        fig.update_layout(
            title='Spor Türlerine Göre Performans Profilleri',
            showlegend=False,
            height=800
        )
        
        fig.write_html('visualizations/radar_chart.html')
        fig.show()
        
    def create_experience_analysis(self):
        """Deneyim analizi grafikleri"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Deneyim Analizi', fontsize=16, fontweight='bold')
        
        # Spor yılı dağılımı
        axes[0, 0].hist(self.data['spor_yili'], bins=15, color='coral', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Spor Deneyimi Dağılımı')
        axes[0, 0].set_xlabel('Spor Yılı')
        axes[0, 0].set_ylabel('Kişi Sayısı')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Yaralanma geçmişi
        injury_counts = self.data['yaralanma_gecmisi'].value_counts()
        axes[0, 1].pie(injury_counts.values, labels=injury_counts.index, autopct='%1.1f%%',
                       colors=['lightgreen', 'yellow', 'orange', 'red'])
        axes[0, 1].set_title('Yaralanma Geçmişi Dağılımı')
        
        # Takım oyunu tercihi
        team_pref = self.data['takım_oyunu_tercihi'].value_counts()
        axes[1, 0].bar(team_pref.index, team_pref.values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 0].set_title('Takım Oyunu Tercihi')
        axes[1, 0].set_xlabel('Tercih')
        axes[1, 0].set_ylabel('Kişi Sayısı')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Spor yılı vs yaş
        axes[1, 1].scatter(self.data['yas'], self.data['spor_yili'], 
                          alpha=0.6, color='purple')
        axes[1, 1].set_title('Yaş vs Spor Deneyimi')
        axes[1, 1].set_xlabel('Yaş')
        axes[1, 1].set_ylabel('Spor Yılı')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/experience_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_comprehensive_dashboard(self):
        """Kapsamlı dashboard oluşturur"""
        # Visualizations klasörünü oluştur
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        print("Kapsamlı veri görselleştirme dashboard'u oluşturuluyor...")
        
        # Tüm analiz grafikleri
        self.create_demographic_analysis()
        self.create_sport_distribution_analysis()
        self.create_performance_analysis()
        self.create_experience_analysis()
        self.create_interactive_scatter_plot()
        self.create_radar_chart()
        
        print("Dashboard oluşturuldu! Grafikler 'visualizations' klasörüne kaydedildi.")
        
    def generate_insights(self):
        """Veri setinden çıkarılan önemli bulgular"""
        print("\n" + "="*60)
        print("VERİ SETİ BULGULARI VE İÇGÖRÜLER")
        print("="*60)
        
        # Temel istatistikler
        print(f"📊 Toplam kişi sayısı: {len(self.data)}")
        print(f"📊 Ortalama yaş: {self.data['yas'].mean():.1f}")
        print(f"📊 En genç: {self.data['yas'].min()}, En yaşlı: {self.data['yas'].max()}")
        
        # Cinsiyet dağılımı
        gender_dist = self.data['cinsiyet'].value_counts(normalize=True) * 100
        print(f"\n👥 Cinsiyet dağılımı:")
        for gender, pct in gender_dist.items():
            print(f"   {gender}: %{pct:.1f}")
        
        # Vücut tipi dağılımı
        body_type_dist = self.data['vucut_tipi'].value_counts(normalize=True) * 100
        print(f"\n🏋️ Vücut tipi dağılımı:")
        for body_type, pct in body_type_dist.items():
            print(f"   {body_type}: %{pct:.1f}")
        
        # En popüler sporlar
        sport_dist = self.data['tavsiye_edilen_spor'].value_counts()
        print(f"\n🏆 En çok tavsiye edilen sporlar:")
        for i, (sport, count) in enumerate(sport_dist.head(5).items(), 1):
            print(f"   {i}. {sport}: {count} kişi (%{count/len(self.data)*100:.1f})")
        
        # Performans ortalamaları
        perf_features = ['hiz', 'kuvvet', 'dayaniklilik', 'esneklik', 'koordinasyon']
        print(f"\n⚡ Ortalama performans skorları:")
        for feature in perf_features:
            avg_score = self.data[feature].mean()
            print(f"   {feature.title()}: {avg_score:.1f}/10")
        
        # Spor deneyimi
        avg_experience = self.data['spor_yili'].mean()
        print(f"\n🎯 Ortalama spor deneyimi: {avg_experience:.1f} yıl")
        
        # Yaralanma oranları
        injury_rates = self.data['yaralanma_gecmisi'].value_counts(normalize=True) * 100
        print(f"\n🏥 Yaralanma geçmişi oranları:")
        for injury, rate in injury_rates.items():
            print(f"   {injury}: %{rate:.1f}")

# Kullanım örneği
if __name__ == "__main__":
    visualizer = SportsDataVisualizer()
    visualizer.load_data()
    visualizer.create_comprehensive_dashboard()
    visualizer.generate_insights() 