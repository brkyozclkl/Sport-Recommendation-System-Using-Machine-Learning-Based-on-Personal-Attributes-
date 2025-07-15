"""
Streamlit Ana Uygulama - Spor Yetenek Tahmin Sistemi
Bu dosya, kullanıcı arayüzünü sağlar.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# Ana proje dizinini path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.feature_config import ALL_FEATURES, TARGET_SPORTS, FEATURE_GROUPS
    from src.data_generator import SportsDataGenerator
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from feature_config import ALL_FEATURES, TARGET_SPORTS, FEATURE_GROUPS
    from data_generator import SportsDataGenerator

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Spor Yetenek Tahmin Sistemi",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .sport-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .sidebar-section {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SportsApp:
    """Streamlit spor yetenek tahmin uygulaması"""
    
    def __init__(self):
        self.data_generator = SportsDataGenerator()
        self.model_data = None
        self.load_model()
        
    def load_model(self):
        """Eğitilmiş modeli yükler"""
        model_path = "models/best_model.pkl"
        if os.path.exists(model_path):
            try:
                self.model_data = joblib.load(model_path)
                st.success("✅ Model başarıyla yüklendi!")
            except Exception as e:
                st.error(f"❌ Model yüklenirken hata: {str(e)}")
        else:
            st.warning("⚠️ Model bulunamadı. Veri üretici ile tahmin yapılacak.")
    
    def render_sidebar(self):
        """Yan bar arayüzünü oluşturur"""
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.title("🏃‍♂️ Spor Yetenek Tahmini")
        st.sidebar.markdown("Kişisel bilgilerinizi girin:")
        
        # Demografik bilgiler
        st.sidebar.subheader("📋 Demografik Bilgiler")
        yas = st.sidebar.slider("Yaş", 12, 50, 25)
        cinsiyet = st.sidebar.selectbox("Cinsiyet", ["Erkek", "Kadın"])
        boy = st.sidebar.slider("Boy (cm)", 140, 220, 170)
        kilo = st.sidebar.slider("Kilo (kg)", 40, 150, 70)
        
        # Vücut tipi
        st.sidebar.subheader("🏋️ Fiziksel Özellikler")
        vucut_tipi = st.sidebar.selectbox("Vücut Tipi", ["Ektomorf", "Mezomorf", "Endomorf"])
        kas_orani = st.sidebar.slider("Kas Oranı (%)", 15, 45, 25)
        yag_orani = st.sidebar.slider("Yağ Oranı (%)", 5, 35, 15)
        
        # Performans özellikleri
        st.sidebar.subheader("⚡ Performans Özellikleri")
        hiz = st.sidebar.slider("Hız (1-10)", 1, 10, 5)
        kuvvet = st.sidebar.slider("Kuvvet (1-10)", 1, 10, 5)
        dayaniklilik = st.sidebar.slider("Dayanıklılık (1-10)", 1, 10, 5)
        esneklik = st.sidebar.slider("Esneklik (1-10)", 1, 10, 5)
        koordinasyon = st.sidebar.slider("Koordinasyon (1-10)", 1, 10, 5)
        
        # Deneyim
        st.sidebar.subheader("🎯 Deneyim")
        spor_yili = st.sidebar.slider("Spor Deneyimi (yıl)", 0, 30, 5)
        takım_oyunu = st.sidebar.selectbox("Takım Oyunu Tercihi", ["Bireysel", "Takım", "Karma"])
        
        # Genetik faktörler
        st.sidebar.subheader("🧬 Genetik Faktörler")
        ailevi_spor = st.sidebar.selectbox("Ailevi Spor Geçmişi", ["Yok", "Var"])
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Tahmin butonu
        if st.sidebar.button("🔮 Spor Tahmini Yap", type="primary"):
            user_data = {
                'yas': yas,
                'cinsiyet': cinsiyet,
                'boy': boy,
                'kilo': kilo,
                'vucut_tipi': vucut_tipi,
                'kas_orani': kas_orani,
                'yag_orani': yag_orani,
                'hiz': hiz,
                'kuvvet': kuvvet,
                'dayaniklilik': dayaniklilik,
                'esneklik': esneklik,
                'koordinasyon': koordinasyon,
                'spor_yili': spor_yili,
                'takım_oyunu_tercihi': takım_oyunu,
                'ailevi_spor_gecmisi': ailevi_spor
            }
            return user_data
        
        return None
    
    def predict_sport(self, user_data):
        """Kullanıcı verisine göre spor tahmini yapar"""
        if self.model_data:
            # Eğitilmiş model ile tahmin
            return self.predict_with_model(user_data)
        else:
            # Veri üretici ile tahmin
            return self.predict_with_generator(user_data)
    
    def predict_with_model(self, user_data):
        """Eğitilmiş model ile tahmin yapar"""
        try:
            # Model verilerini al
            model = self.model_data['model']
            scaler = self.model_data['scaler']
            label_encoder = self.model_data['label_encoder']
            feature_names = self.model_data['feature_names']
            
            # Kullanıcı verisini model formatına dönüştür
            full_data = self.prepare_user_data_for_model(user_data)
            
            # Kategorik verileri encode et
            user_df = pd.DataFrame([full_data])
            categorical_columns = user_df.select_dtypes(include=['object']).columns
            user_encoded = pd.get_dummies(user_df, columns=categorical_columns, drop_first=True)
            
            # Sütun isimlerini temizle
            import re
            user_encoded.columns = [re.sub(r'[^\w\s]', '_', col).replace(' ', '_') 
                                   for col in user_encoded.columns]
            
            # Model özelliklerini eşleştir
            for feature in feature_names:
                if feature not in user_encoded.columns:
                    user_encoded[feature] = 0
            
            # Sadece model özelliklerini al
            user_encoded = user_encoded[feature_names]
            
            # Model tipine göre scaling
            if self.model_data['model_name'] in ['SVM', 'Neural Network']:
                user_scaled = scaler.transform(user_encoded)
                predictions = model.predict(user_scaled)
                probabilities = model.predict_proba(user_scaled)
            else:
                predictions = model.predict(user_encoded)
                probabilities = model.predict_proba(user_encoded)
            
            # Tahmin sonuçlarını çözümle
            predicted_sport = label_encoder.inverse_transform(predictions)[0]
            
            # Tüm sporlar için olasılık skorları
            sport_scores = {}
            for i, sport in enumerate(label_encoder.classes_):
                sport_scores[sport] = probabilities[0][i] * 100
            
            return predicted_sport, sport_scores
            
        except Exception as e:
            st.error(f"Model tahmin hatası: {str(e)}")
            # Hata durumunda veri üretici ile tahmin yap
            return self.predict_with_generator(user_data)
    
    def prepare_user_data_for_model(self, user_data):
        """Kullanıcı verisini model için hazırlar"""
        # Eksik alanları varsayılan değerlerle doldur
        full_data = {
            'yas': user_data['yas'],
            'cinsiyet': user_data['cinsiyet'],
            'boy': user_data['boy'],
            'kilo': user_data['kilo'],
            'vucut_tipi': user_data['vucut_tipi'],
            'kas_orani': user_data['kas_orani'],
            'yag_orani': user_data['yag_orani'],
            'hiz': user_data['hiz'],
            'kuvvet': user_data['kuvvet'],
            'dayaniklilik': user_data['dayaniklilik'],
            'esneklik': user_data['esneklik'],
            'koordinasyon': user_data['koordinasyon'],
            'spor_yili': user_data['spor_yili'],
            'takım_oyunu_tercihi': user_data['takım_oyunu_tercihi'],
            'ailevi_spor_gecmisi': user_data['ailevi_spor_gecmisi'],
            # Varsayılan değerler
            'bmi': user_data['kilo'] / ((user_data['boy'] / 100) ** 2),
            'kemik_yogunlugu': 'Orta',
            'denge': 5,
            'reaksiyon_hizi': 5,
            'anne_spor_durumu': 'Aktif',
            'baba_spor_durumu': 'Aktif',
            'dominant_el': 'Sağ',
            'onceki_sporlar': "['Futbol']",
            'en_basarili_spor': 'Futbol',
            'yaralanma_gecmisi': 'Yok',
            'stres_toleransi': 5,
            'yarışma_tutkusu': 5,
            'konsantrasyon': 5,
            'coğrafi_konum': 'Şehir',
            'ekonomik_durum': 'Orta',
            'tesis_erisimi': 'Orta'
        }
        
        return full_data
    
    def predict_with_generator(self, user_data):
        """Veri üretici ile tahmin yapar"""
        # Eksik alanları varsayılan değerlerle doldur
        full_data = {
            'yas': user_data['yas'],
            'cinsiyet': user_data['cinsiyet'],
            'boy': user_data['boy'],
            'kilo': user_data['kilo'],
            'vucut_tipi': user_data['vucut_tipi'],
            'kas_orani': user_data['kas_orani'],
            'yag_orani': user_data['yag_orani'],
            'hiz': user_data['hiz'],
            'kuvvet': user_data['kuvvet'],
            'dayaniklilik': user_data['dayaniklilik'],
            'esneklik': user_data['esneklik'],
            'koordinasyon': user_data['koordinasyon'],
            'spor_yili': user_data['spor_yili'],
            'takım_oyunu_tercihi': user_data['takım_oyunu_tercihi'],
            'ailevi_spor_gecmisi': user_data['ailevi_spor_gecmisi'],
            # Varsayılan değerler
            'bmi': user_data['kilo'] / ((user_data['boy'] / 100) ** 2),
            'kemik_yogunlugu': 'Orta',
            'denge': 5,
            'reaksiyon_hizi': 5,
            'anne_spor_durumu': 'Aktif',
            'baba_spor_durumu': 'Aktif',
            'dominant_el': 'Sağ',
            'onceki_sporlar': ['Futbol'],
            'en_basarili_spor': 'Futbol',
            'yaralanma_gecmisi': 'Yok',
            'stres_toleransi': 5,
            'yarışma_tutkusu': 5,
            'konsantrasyon': 5,
            'coğrafi_konum': 'Şehir',
            'ekonomik_durum': 'Orta',
            'tesis_erisimi': 'Orta'
        }
        
        # Spor uyumluluk skorlarını hesapla
        sport_scores = self.data_generator._calculate_sport_compatibility(full_data)
        
        # En iyi sporu bul
        best_sport = max(sport_scores, key=sport_scores.get)
        
        return best_sport, sport_scores
    
    def render_prediction_results(self, user_data):
        """Tahmin sonuçlarını gösterir"""
        st.markdown('<div class="section-title">🎯 Tahmin Sonuçları</div>', unsafe_allow_html=True)
        
        # Tahmin yap
        best_sport, sport_scores = self.predict_sport(user_data)
        
        # Ana tahmin sonucu
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="sport-card">
                <h2>🏆 Size En Uygun Spor: {best_sport}</h2>
                <p>Uyumluluk Skoru: {sport_scores[best_sport]:.1f}/100</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # BMI hesaplama
            bmi = user_data['kilo'] / ((user_data['boy'] / 100) ** 2)
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 BMI Değeriniz</h4>
                <h2>{bmi:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Tüm sporların skorları
        st.markdown('<div class="section-title">📊 Tüm Spor Skorları</div>', unsafe_allow_html=True)
        
        # Skorları sırala
        sorted_scores = sorted(sport_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Bar chart
        sports, scores = zip(*sorted_scores)
        fig = px.bar(
            x=list(scores),
            y=list(sports),
            orientation='h',
            title='Spor Uyumluluk Skorları',
            labels={'x': 'Uyumluluk Skoru', 'y': 'Spor'},
            color=list(scores),
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart
        st.markdown('<div class="section-title">🕸️ Performans Profili</div>', unsafe_allow_html=True)
        
        performance_features = ['hiz', 'kuvvet', 'dayaniklilik', 'esneklik', 'koordinasyon']
        performance_values = [user_data.get(feature, 5) for feature in performance_features]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=performance_values,
            theta=[feature.title() for feature in performance_features],
            fill='toself',
            name='Performans Profili',
            line_color='rgb(32, 201, 151)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10])
            ),
            showlegend=False,
            title="Kişisel Performans Profili",
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Öneriler
        st.markdown('<div class="section-title">💡 Kişiselleştirilmiş Öneriler</div>', unsafe_allow_html=True)
        
        recommendations = self.generate_recommendations(user_data, best_sport, sport_scores)
        
        for rec in recommendations:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{rec['title']}</h4>
                <p>{rec['content']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def generate_recommendations(self, user_data, best_sport, sport_scores):
        """Kişiselleştirilmiş öneriler üretir"""
        recommendations = []
        
        # En iyi spor için öneriler
        sport_info = TARGET_SPORTS.get(best_sport, {})
        key_features = sport_info.get('key_features', [])
        
        recommendations.append({
            'title': f'🎯 {best_sport} İçin Öneriler',
            'content': f'Bu spor için önemli özellikler: {", ".join(key_features)}. '
                      f'Bu alanlarda kendinizi geliştirmeye odaklanın.'
        })
        
        # Performans önerileri
        performance_features = ['hiz', 'kuvvet', 'dayaniklilik', 'esneklik', 'koordinasyon']
        weak_areas = [f for f in performance_features if user_data.get(f, 5) < 5]
        
        if weak_areas:
            recommendations.append({
                'title': '💪 Geliştirilmesi Gereken Alanlar',
                'content': f'Şu alanlarda antrenman yaparak kendinizi geliştirebilirsiniz: '
                          f'{", ".join(weak_areas)}.'
            })
        
        # Alternatif sporlar
        alternative_sports = [sport for sport, score in sorted(sport_scores.items(), 
                                                               key=lambda x: x[1], 
                                                               reverse=True)[1:4]]
        
        recommendations.append({
            'title': '🏃‍♀️ Alternatif Sporlar',
            'content': f'Ayrıca şu sporları da deneyebilirsiniz: {", ".join(alternative_sports)}.'
        })
        
        return recommendations
    
    def render_data_analysis(self):
        """Veri analizi sayfasını oluşturur"""
        st.markdown('<div class="section-title">📊 Veri Analizi</div>', unsafe_allow_html=True)
        
        # Veri setini yükle
        try:
            data = pd.read_csv("data/sporcu_dataset.csv")
            
            # Temel istatistikler
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam Kişi", len(data))
            with col2:
                st.metric("Ortalama Yaş", f"{data['yas'].mean():.1f}")
            with col3:
                st.metric("Spor Türü", len(data['tavsiye_edilen_spor'].unique()))
            with col4:
                st.metric("Özellik Sayısı", len(data.columns))
            
            # Spor dağılımı
            st.subheader("🏆 Spor Dağılımı")
            sport_counts = data['tavsiye_edilen_spor'].value_counts()
            fig_pie = px.pie(
                values=sport_counts.values,
                names=sport_counts.index,
                title="Tavsiye Edilen Spor Dağılımı"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Yaş dağılımı
            st.subheader("👥 Yaş Dağılımı")
            fig_hist = px.histogram(
                data, 
                x='yas', 
                nbins=20,
                title="Yaş Dağılımı",
                labels={'yas': 'Yaş', 'count': 'Kişi Sayısı'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Performans korelasyonu
            st.subheader("🔗 Performans Korelasyonu")
            perf_features = ['hiz', 'kuvvet', 'dayaniklilik', 'esneklik', 'koordinasyon']
            corr_matrix = data[perf_features].corr()
            
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Performans Özellikleri Korelasyon Matrisi"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        except FileNotFoundError:
            st.error("Veri seti bulunamadı. Lütfen önce veri setini oluşturun.")
    
    def run(self):
        """Ana uygulamayı çalıştırır"""
        # Başlık
        st.markdown('<div class="main-title">🏃‍♂️ Spor Yetenek Tahmin Sistemi</div>', 
                   unsafe_allow_html=True)
        
        # Sekme seçenekleri
        tab1, tab2, tab3 = st.tabs(["🎯 Tahmin", "📊 Veri Analizi", "ℹ️ Hakkında"])
        
        with tab1:
            # Yan bar
            user_data = self.render_sidebar()
            
            if user_data:
                # Tahmin sonuçlarını göster
                self.render_prediction_results(user_data)
            else:
                # Hoş geldiniz mesajı
                st.markdown("""
                ### Hoş Geldiniz! 🎉
                
                Bu sistem, kişisel özelliklerinize dayalı olarak size en uygun sporu tahmin eder.
                
                **Nasıl Kullanılır:**
                1. Sol taraftaki formu doldurun
                2. "Spor Tahmini Yap" butonuna tıklayın
                3. Sonuçları görüntüleyin ve önerileri takip edin
                
                **Özellikler:**
                - 🎯 Kişiselleştirilmiş spor önerileri
                - 📊 Detaylı performans analizi
                - 💡 Gelişim önerileri
                - 🏆 10 farklı spor kategorisi
                """)
        
        with tab2:
            self.render_data_analysis()
        
        with tab3:
            st.markdown("""
            ### Spor Yetenek Tahmin Sistemi Hakkında
            
            Bu sistem, makine öğrenmesi algoritmaları kullanarak bireylerin fiziksel ve demografik 
            özelliklerine dayalı spor önerileri sunar.
            
            **Kullanılan Teknolojiler:**
            - 🐍 Python
            - 🧠 Scikit-learn, XGBoost, LightGBM
            - 📊 Streamlit, Plotly
            - 🎨 Pandas, NumPy
            
            **Değerlendirilen Özellikler:**
            - 📋 Demografik bilgiler (yaş, cinsiyet, boy, kilo)
            - 🏋️ Fiziksel özellikler (vücut tipi, kas oranı)
            - ⚡ Performans yetenekleri (hız, kuvvet, dayanıklılık)
            - 🎯 Spor deneyimi ve tercihleri
            - 🧬 Genetik faktörler
            
            **Desteklenen Sporlar:**
            - 🏃‍♀️ Koşu/Atletizm
            - ⚽ Futbol
            - 🏀 Basketbol
            - 🏐 Voleybol
            - 🎾 Tenis
            - 🏊‍♀️ Yüzme
            - 🤸‍♀️ Jimnastik
            - 🥊 Boks
            - 🤼‍♀️ Güreş
            - 🚴‍♀️ Bisiklet
            
            **Geliştirici:** AI Spor Analiz Sistemi
            **Sürüm:** 1.0.0
            """)

# Uygulamayı çalıştır
if __name__ == "__main__":
    app = SportsApp()
    app.run() 