"""
Makine Öğrenmesi Modelleri - Spor Yetenek Tahmin Sistemi
Bu dosya, farklı ML modellerini eğitir ve karşılaştırır.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class SportsModelTrainer:
    """Spor yetenek tahmin modelleri eğitici sınıfı"""
    
    def __init__(self, data_path: str = "data/sporcu_dataset_500.csv"):
        """
        Model eğitici sınıfını başlatır
        
        Args:
            data_path: Veri seti dosya yolu
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Veri setini yükler ve ön işlemden geçirir"""
        print("Veri seti yükleniyor...")
        self.data = pd.read_csv(self.data_path)
        print(f"Veri boyutu: {self.data.shape}")
        
        # Hedef değişken (tavsiye_edilen_spor) ve özellikler
        target_column = 'tavsiye_edilen_spor'
        feature_columns = [col for col in self.data.columns 
                          if col != target_column and not col.startswith('skor_')]
        
        # Eksik değer kontrolü
        print(f"Eksik değer sayısı: {self.data.isnull().sum().sum()}")
        
        # Kategorik değişkenleri sayısal hale getir
        X = self.data[feature_columns].copy()
        
        # Kategorik sütunları belirle
        categorical_columns = X.select_dtypes(include=['object']).columns
        print(f"Kategorik sütunlar: {list(categorical_columns)}")
        
        # One-hot encoding
        X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        # Sütun isimlerini temizle (XGBoost ve LightGBM için)
        import re
        X_encoded.columns = [re.sub(r'[^\w\s]', '_', col).replace(' ', '_') 
                           for col in X_encoded.columns]
        
        # Özellik matrisi ve hedef değişken
        self.X = X_encoded
        self.y = self.label_encoder.fit_transform(self.data[target_column])
        
        print(f"Özellik sayısı: {self.X.shape[1]}")
        print(f"Sınıf sayısı: {len(np.unique(self.y))}")
        print(f"Sınıf dağılımı: {dict(zip(self.label_encoder.classes_, np.bincount(self.y)))}")
        
        # Veriyi eğitim ve test setlerine ayır
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Standartlaştırma
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Eğitim seti boyutu: {self.X_train.shape}")
        print(f"Test seti boyutu: {self.X_test.shape}")
        
    def initialize_models(self):
        """Makine öğrenmesi modellerini başlatır"""
        print("\nModeller başlatılıyor...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        print(f"Toplam {len(self.models)} model başlatıldı")
        
    def train_and_evaluate_models(self):
        """Tüm modelleri eğitir ve değerlendirir"""
        print("\nModel eğitimi başlıyor...")
        
        for name, model in self.models.items():
            print(f"\n{name} modeli eğitiliyor...")
            
            # Modeli eğit
            if name in ['SVM', 'Neural Network']:
                # Standartlaştırılmış veri kullan
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=5, scoring='accuracy')
            else:
                # Orijinal veri kullan
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=5, scoring='accuracy')
            
            # Performans metrikleri
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Sonuçları kaydet
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(
                    self.y_test, y_pred, 
                    labels=np.unique(np.concatenate([self.y_test, y_pred])),
                    target_names=[self.label_encoder.classes_[i] for i in np.unique(np.concatenate([self.y_test, y_pred]))],
                    output_dict=True,
                    zero_division=0
                )
            }
            
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
    def get_feature_importance(self):
        """Özellik önem skorlarını hesaplar"""
        print("\nÖzellik önem skorları hesaplanıyor...")
        
        feature_importance = {}
        
        for name, result in self.results.items():
            model = result['model']
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based modeller için
                importance = model.feature_importances_
                feature_importance[name] = dict(zip(self.X.columns, importance))
            elif hasattr(model, 'coef_'):
                # Linear modeller için
                importance = np.abs(model.coef_[0])
                feature_importance[name] = dict(zip(self.X.columns, importance))
            else:
                feature_importance[name] = {}
                
        return feature_importance
        
    def save_best_model(self, save_path: str = "models/best_model.pkl"):
        """En iyi modeli kaydeder"""
        # En iyi modeli bul
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_model_name]['model']
        
        # Modeli kaydet
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': list(self.X.columns),
            'model_name': best_model_name,
            'accuracy': self.results[best_model_name]['accuracy']
        }
        
        joblib.dump(model_data, save_path)
        print(f"\nEn iyi model kaydedildi: {best_model_name}")
        print(f"Dosya yolu: {save_path}")
        print(f"Accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        
    def print_results_summary(self):
        """Sonuçları özetler"""
        print("\n" + "="*50)
        print("MODEL KARŞILAŞTIRMA SONUÇLARI")
        print("="*50)
        
        # Sonuçları accuracy'ye göre sırala
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['accuracy'], 
                              reverse=True)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"\n{i}. {name}")
            print(f"   Test Accuracy: {result['accuracy']:.4f}")
            print(f"   CV Accuracy: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
            
            # Precision, Recall, F1-Score
            report = result['classification_report']
            print(f"   Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
            print(f"   Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
            
    def run_full_pipeline(self):
        """Tam eğitim sürecini çalıştırır"""
        print("Spor Yetenek Tahmin Sistemi - Model Eğitimi")
        print("="*50)
        
        # Veri yükleme ve ön işleme
        self.load_and_preprocess_data()
        
        # Modelleri başlat
        self.initialize_models()
        
        # Modelleri eğit ve değerlendir
        self.train_and_evaluate_models()
        
        # Sonuçları yazdır
        self.print_results_summary()
        
        # Özellik önem skorları
        feature_importance = self.get_feature_importance()
        
        # En iyi modeli kaydet
        self.save_best_model()
        
        return self.results, feature_importance

# Kullanım örneği
if __name__ == "__main__":
    trainer = SportsModelTrainer()
    results, feature_importance = trainer.run_full_pipeline()
    
    print("\n" + "="*50)
    print("EĞİTİM TAMAMLANDI!")
    print("="*50) 