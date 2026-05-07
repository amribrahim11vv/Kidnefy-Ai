"""
Data Loader Module
Handles loading, merging, and preprocessing of kidney disease datasets.
يقوم هذا الملف بتحميل ودمج ومعالجة مجموعات بيانات أمراض الكلى.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing.feature_engineering import FeatureEngineer


# ======================================================================
# ضع أسماء ملفات الداتاسيت هنا
# Place your dataset filenames here
# ======================================================================
# ======================================================================
CKD_DATASET_FILE = "kidney_disease.csv"                    # ← اسم ملف داتاسيت CKD (مثال: "kidney_disease.csv")
DIABETIC_NEPHROPATHY_FILE = "Diabetic_Nephropathy_v1.xlsx"           # ← اسم ملف داتاسيت Diabetic Nephropathy (مثال: "dn_data.xlsx")
DIABETIC_NEPHROPATHY_2_FILE = "diabetic_nephropathy2_dataset.csv"      # ← Dataset with HbA1c, eGFR, UACR
DIABETES_PREDICTION_FILE = "diabetes_prediction_dataset.csv"            # ← اسم ملف داتاسيت Diabetes Prediction (مثال: "diabetes_prediction_dataset.csv")
# ======================================================================


class DataLoader:
    """Load, merge, and preprocess kidney disease datasets."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_names: list = []
        
    def load_ckd_dataset(self) -> pd.DataFrame:
        """Load the Chronic Kidney Disease dataset."""
        if CKD_DATASET_FILE:
            filepath = self.data_dir / CKD_DATASET_FILE
            if filepath.exists():
                if filepath.suffix == '.xlsx':
                    df = pd.read_excel(filepath)
                else:
                    df = pd.read_csv(filepath)
                print(f"[OK] Loaded CKD dataset from {filepath}")
                return df
            else:
                raise FileNotFoundError(f"CKD dataset file not found: {filepath}")
        
        raise FileNotFoundError(
            "❌ CKD dataset filename is empty!\n"
            "   Please set CKD_DATASET_FILE in data_loader.py\n"
            "   Example: CKD_DATASET_FILE = \"kidney_disease.csv\""
        )
    
    def load_diabetic_nephropathy_dataset(self) -> pd.DataFrame:
        """Load the Diabetic Nephropathy dataset."""
        if DIABETIC_NEPHROPATHY_FILE:
            filepath = self.data_dir / DIABETIC_NEPHROPATHY_FILE
            if filepath.exists():
                if filepath.suffix == '.xlsx':
                    df = pd.read_excel(filepath)
                else:
                    df = pd.read_csv(filepath)
                print(f"[OK] Loaded Diabetic Nephropathy dataset from {filepath}")
                return df
            else:
                raise FileNotFoundError(f"Diabetic Nephropathy dataset file not found: {filepath}")
        
        raise FileNotFoundError(
            "   Please set DIABETIC_NEPHROPATHY_FILE in data_loader.py\n"
            "   Example: DIABETIC_NEPHROPATHY_FILE = \"Diabetic_Nephropathy_v1.xlsx\""
        )

    def load_dn2_dataset(self) -> pd.DataFrame:
        """Load the new Diabetic Nephropathy dataset (dataset 2)."""
        if DIABETIC_NEPHROPATHY_2_FILE:
            filepath = self.data_dir / DIABETIC_NEPHROPATHY_2_FILE
            if filepath.exists():
                df = pd.read_csv(filepath)
                print(f"[OK] Loaded Diabetic Nephropathy 2 dataset from {filepath}")
                return df
            else:
                # Optional: don't fail hard if this specific file is missing, just warn
                print(f"[WARN] Diabetic Nephropathy 2 dataset file not found: {filepath}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def merge_datasets(self, df_ckd: pd.DataFrame, df_dn: pd.DataFrame, df_dn2: pd.DataFrame = None) -> pd.DataFrame:
        """
        Merge CKD, Diabetic Nephropathy 1, and Diabetic Nephropathy 2 datasets.
        دمج الداتاسيتات الثلاثة في داتاسيت واحد.
        
        - Maps columns to standard names.
        - Adds a 'source' column.
        - Concatenates DataFrames.
        
        Returns:
            Merged DataFrame
        """
        df_ckd = df_ckd.copy()
        df_dn = df_dn.copy()
        
        # Add source column
        df_ckd['source'] = 'ckd'
        df_dn['source'] = 'diabetic_nephropathy'
        
        dfs_to_merge = [df_ckd, df_dn]
        
        if df_dn2 is not None and not df_dn2.empty:
            df_dn2 = df_dn2.copy()
            df_dn2['source'] = 'diabetic_nephropathy_2'
            
            # Map columns in df_dn2 to match training features
            # serum_creatinine -> sc
            # BUN -> bu
            # blood_pressure_systolic -> bp
            # blood_glucose -> bgr
            # hypertension -> htn
            # CKD_stage -> classification (mapping needed later or treat as separate target)
            # DN_present -> dm (maybe?)
            
            column_mapping = {
                'serum_creatinine': 'sc',
                'BUN': 'bu',
                'blood_pressure_systolic': 'bp',
                'blood_glucose': 'bgr',
                'hypertension': 'htn',
                # New features renames
                'blood_pressure_diastolic': 'bp_dia',
                'albumin': 'serum_albumin',  # Distinguish from urine albumin 'al'
                'diabetes_duration_years': 'diabetes_duration',
                'HbA1c': 'hba1c',
                'eGFR': 'egfr',
                'UACR': 'uacr',
                'BMI': 'bmi',
                'uric_acid': 'uric_acid',
                'smoking': 'smoking',
                'dyslipidemia': 'dyslipidemia',
                'diabetes_type': 'diabetes_type',
                'gender': 'gender',
                'CKD_stage': 'ckd_stage',
                'DN_present': 'dn_present',
                'risk_level': 'risk_level'
            }
            df_dn2 = df_dn2.rename(columns=column_mapping)
            
            # Normalize htn (Yes/No -> 1/0)
            if 'htn' in df_dn2.columns:
                df_dn2['htn'] = df_dn2['htn'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
            
            # Normalize dn_present/dm (column was renamed from DN_present -> dn_present)
            if 'dn_present' in df_dn2.columns:
                df_dn2['dm'] = df_dn2['dn_present'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
            
            # Normalize CKD_stage to classification if needed
            # (column was renamed from CKD_stage -> ckd_stage)
            if 'ckd_stage' in df_dn2.columns:
                # Map stages to ckd/notckd
                # Stage 1, 2, 3a, 3b, 4, 5 -> usually CKD if GFR < 60 or markers present
                # But in UCI dataset, it is binary.
                # Let's add a classification column for compatibility
                df_dn2['classification'] = 'ckd' # Assume all in this dataset are patients or controls?
                # Actually checking dataset: it has 'risk_level'.
                # Let's look at eGFR. If eGFR >= 90 and no albuminuria -> likely not ckd
                # But simplify: row 6 is '1', row 7 is '1', risk 'Low'.
                # Let's infer classification.
                # If risk_level is 'Low' -> notckd?
                # Let's do: 'ckd' if CKD_stage != '1' or UACR > 30 else 'notckd'
                # Or just mark all as 'ckd' if they are from a "Diabetic Nephropathy" dataset?
                # Actually the dataset name suggests they are patients.
                # But risk_level has 'Low'.
                # Let's map based on CKD_stage.
                # 3a, 3b, 4, 5 -> ckd. 1, 2 -> check albuminuria.
                # For simplicity in this merge, let's allow nan classification and let preprocessing handle it or fill it.
                pass

            dfs_to_merge.append(df_dn2)
            print(f"[OK] Included Diabetic Nephropathy 2 dataset ({len(df_dn2)} rows)")

        # Find common columns
        print("Merging datasets...")
        merged = pd.concat(dfs_to_merge, axis=0, ignore_index=True)
        
        print(f" Merged dataset shape: {merged.shape}")
        print(f"   - CKD rows: {len(df_ckd)}")
        print(f"   - Diabetic Nephropathy rows: {len(df_dn)}")
        print(f"   - Total rows: {len(merged)}")
        
        return merged
    
    def _clean_raw_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str, list, list]:
        """
        Phase 1: Clean raw data (safe to do before split).
        - Drop leakage/id columns
        - Replace ? with NaN
        - Identify column types
        
        Returns:
            df, target_col, numerical_cols, categorical_cols
        """
        df = df.copy()
        
        # Drop non-feature columns
        cols_to_drop = ['id', 'source']
        
        # [ALERT] ANTI-LEAKAGE: Drop columns that directly reveal the target
        # uacr = Urine Albumin-Creatinine Ratio = ACR
        # KDIGO defines CKD as: eGFR < 60 OR ACR >= 30 mg/g
        # → uacr directly satisfies the DIAGNOSTIC CRITERION for CKD.
        #   Including it lets the model "look up the answer" rather than predicting risk.
        # serum_albumin: Low albumin is a direct consequence of severe proteinuria (CKD marker).
        leakage_cols = [
            'ckd_stage', 'dn_present', 'risk_level', 'eGFR', 'egfr', 'CKD_stage',
            'uacr',           # ACR — KDIGO diagnostic criterion (eGFR<60 OR ACR≥30)
            'serum_albumin',  # Low albumin = consequence of CKD-level proteinuria
        ]
        cols_to_drop.extend(leakage_cols)
        
        # [ALERT] DROP CKD SYMPTOM FEATURES (these cause 99.9% accuracy)
        # These features are CKD SYMPTOMS/DIAGNOSTIC CRITERIA, not early predictors.
        # Including them means the model is reading symptoms, not predicting disease.
        # 
        # Medical rationale:
        #   hemo/pcv/rc  → Anemia indicators (anemia is a RESULT of CKD, not a predictor)
        #   al           → Urine albumin (0-5 scale) — this IS a diagnostic criterion for CKD
        #   rbc/pc/pcc   → Urine sediment analysis — found AFTER CKD develops
        #   ba           → Bacteria in urine — infection indicator
        #   pe           → Pedal edema — CKD symptom (fluid retention)
        #   ane          → Anemia flag — same info as hemoglobin (CKD symptom)
        #   appet        → Poor appetite — late-stage CKD symptom (uremia)
        #   wc           → White blood cell count — infection/inflammation marker
        #   sg           → Specific gravity — directly related to kidney concentrating ability
        #
        # [ALERT] [UPDATE] DROP DIRECT BIOMARKERS (sc, bu)
        #   sc           → Serum Creatinine: The exact metric doctors use to calculate eGFR and diagnose CKD.
        #   bu           → Blood Urea: Rises directly with kidney failure.
        #   By dropping these, we force the model to PREDICT based on risk factors (age, bp, sugar) 
        #   rather than just applying the standard medical formula.
        #
        # What remains (early screening risk factors):
        #   age, bp, bgr, sod, pot, su, htn, dm, cad + engineered features
        symptom_cols = [
            'hemo', 'pcv', 'rc', 'wc',    # Blood count (anemia = CKD symptom)
            'al',                            # Urine albumin (IS the diagnostic criterion)
            'rbc', 'pc', 'pcc', 'ba',       # Urine sediment (CKD indicators)
            'pe', 'ane', 'appet',            # CKD symptoms (edema, anemia, appetite)
            'sg',                            # Specific gravity (kidney function indicator)
            'sc', 'bu'                       # Direct diagnosis biomarkers (Target Leakage)
        ]
        cols_to_drop.extend(symptom_cols)
        print(f"   [ALERT] Dropping {len(symptom_cols)} CKD diagnostic/symptom features for realistic early prediction")
        
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])

        
        # Replace '?' and 'ckd\t' with NaN
        df = df.replace(['?', '\t?', 'ckd\t', 'notckd\t'], 
                        [np.nan, np.nan, 'ckd', 'notckd'])
        
        # Identify target column
        target_col = None
        for possible_target in ['classification', 'class', 'target', 'label']:
            if possible_target in df.columns:
                target_col = possible_target
                break
        if target_col is None:
            target_col = df.columns[-1]

        # Normalize common target strings and drop missing/unknown labels.
        # IMPORTANT: We must NOT allow missing labels to become a separate class like "nan".
        raw_target = df[target_col]
        # Keep NaN as NaN (don't convert to "nan" string yet)
        if pd.api.types.is_object_dtype(raw_target) or pd.api.types.is_string_dtype(raw_target):
            t = raw_target.astype(str).str.strip().str.lower()
            # Map common variants
            t = t.replace(
                {
                    "ckd\t": "ckd",
                    "notckd\t": "notckd",
                    "not ckd": "notckd",
                    "no ckd": "notckd",
                    "normal": "notckd",
                    "none": np.nan,
                    "null": np.nan,
                    "nan": np.nan,
                    "?": np.nan,
                    "unknown": np.nan,
                    "": np.nan,
                }
            )
            # Restore real NaNs where appropriate
            df[target_col] = t

        # Drop rows with missing target labels (cannot be used for supervised training)
        before = len(df)
        df = df.dropna(subset=[target_col])
        dropped = before - len(df)
        if dropped:
            print(f"   [WARN] Dropped {dropped} rows with missing target label in '{target_col}'")
        
        # Encode target variable BEFORE split (safe - it's the label, not a feature)
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col].astype(str).str.strip())
        self.label_encoders['target'] = le_target
        
        # Separate feature columns by type
        feature_cols = [c for c in df.columns if c != target_col]
        
        categorical_cols = []
        numerical_cols = []
        
        for col in feature_cols:
            converted = pd.to_numeric(df[col], errors='coerce')
            non_null_ratio = converted.notna().sum() / len(df)
            
            if non_null_ratio > 0.5:
                numerical_cols.append(col)
                df[col] = converted
            else:
                categorical_cols.append(col)
        
        print(f"   Numerical features: {len(numerical_cols)}")
        print(f"   Categorical features: {len(categorical_cols)}")
        print(f"   Target column: {target_col}")
        
        return df, target_col, numerical_cols, categorical_cols
    
    def _preprocess_split(
        self, 
        df_train: pd.DataFrame, 
        df_val: pd.DataFrame, 
        df_test: pd.DataFrame,
        numerical_cols: list, 
        categorical_cols: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Phase 2: Preprocess AFTER split.
        
        [ALERT] CRITICAL: All imputers/encoders are FIT on train ONLY,
        then TRANSFORM val and test. This prevents preprocessing leakage.
        """
        # ─── Categorical: fill missing with mode from TRAIN only ───
        cat_fill_values = {}
        for col in categorical_cols:
            if col in df_train.columns:
                mode_val = df_train[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else 'unknown'
                cat_fill_values[col] = fill_val
                df_train[col] = df_train[col].fillna(fill_val)
                df_val[col] = df_val[col].fillna(fill_val)
                df_test[col] = df_test[col].fillna(fill_val)
        
        # ─── Numerical: KNN Imputer FIT on train ONLY ───
        if numerical_cols:
            imputer = KNNImputer(n_neighbors=5)
            # FIT on train
            df_train[numerical_cols] = imputer.fit_transform(df_train[numerical_cols])
            # TRANSFORM val and test
            df_val[numerical_cols] = imputer.transform(df_val[numerical_cols])
            df_test[numerical_cols] = imputer.transform(df_test[numerical_cols])
            self._imputer = imputer
        
        # ─── Categorical: LabelEncoder FIT on train ONLY ───
        for col in categorical_cols:
            if col in df_train.columns:
                le = LabelEncoder()
                # Fit on train
                le.fit(df_train[col].astype(str))
                self.label_encoders[col] = le
                # Transform all sets
                df_train[col] = le.transform(df_train[col].astype(str))
                # Handle unseen labels in val/test
                for df_set in [df_val, df_test]:
                    known = set(le.classes_)
                    df_set[col] = df_set[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df_set[col] = le.transform(df_set[col])
        
        # ─── Feature Engineering: applied independently (no leakage) ───
        fe = FeatureEngineer()
        df_train = fe.create_categorical_bins(df_train)
        df_val = fe.create_categorical_bins(df_val)
        df_test = fe.create_categorical_bins(df_test)
        
        # Align columns (some bins might create different dummies)
        all_cols = df_train.columns.tolist()
        for df_set in [df_val, df_test]:
            for col in all_cols:
                if col not in df_set.columns:
                    df_set[col] = 0
        df_val = df_val[all_cols]
        df_test = df_test[all_cols]
        
        return df_train, df_val, df_test
    
    def load_and_prepare_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        feature_selection: bool = True,
        use_smote: bool = False,
        ckd_only: bool = False   # True = use ONLY real UCI CKD data (realistic 83-90% acc)
                                  # False = merge all datasets (higher acc but synthetic data)
    ) -> Tuple:
        """
        Complete LEAKAGE-FREE pipeline with anti-overfitting measures:
          1. Load & merge datasets
          2. Clean raw data (safe ops only)
          3. Split into Train/Val/Test (RAW data)
          4. Preprocess each set (fit on train ONLY)
          5. Scale features (fit on train ONLY)
          6. Feature Selection (fit on train ONLY) — reduces noise features
          7. SMOTE (on train ONLY, optional) — handles class imbalance
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names
        """
        print("=" * 60)
        print("Loading and Merging Datasets...")
        print("=" * 60)
        
        # Load datasets
        df_ckd = self.load_ckd_dataset()

        if ckd_only:
            # ── REALISTIC MODE: UCI CKD only (400 real clinical records) ──
            # Accuracy expected: 83-90%  (real-world generalizable)
            print("[INFO] ckd_only=True → training on real UCI CKD dataset only (realistic accuracy)")
            df_merged = df_ckd
        else:
            # ── COMBINED MODE: all datasets merged (synthetic data included) ──
            # Accuracy expected: 95-99%  (inflated by synthetic patterns)
            df_dn  = self.load_diabetic_nephropathy_dataset()
            df_dn2 = self.load_dn2_dataset()
            df_merged = self.merge_datasets(df_ckd, df_dn, df_dn2)

        # Phase 1: Clean raw data (safe to do on full dataset)
        print("\nCleaning raw data...")
        df_clean, target_col, numerical_cols, categorical_cols = self._clean_raw_data(df_merged)
        
        # ═══════════════════════════════════════════════════════
        # [ALERT] SPLIT FIRST — before any imputation or encoding!
        # ═══════════════════════════════════════════════════════
        print("\nSplitting data BEFORE preprocessing (anti-leakage)...")
        
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Step 1: Split off Test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Step 2: Split remaining into Train (60%) and Val (20%)
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=y_temp
        )
        
        print(f"   Raw split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Phase 2: Preprocess AFTER split (fit on train ONLY)
        print("   Preprocessing (fit on train only, transform val/test)...")
        
        # Keep numerical/categorical cols that exist in features
        num_cols = [c for c in numerical_cols if c in X_train.columns]
        cat_cols = [c for c in categorical_cols if c in X_train.columns]
        
        X_train, X_val, X_test = self._preprocess_split(
            X_train.copy(), X_val.copy(), X_test.copy(),
            num_cols, cat_cols
        )
        
        # Store feature names (after feature engineering)
        self.feature_names = X_train.columns.tolist()
        
        # Phase 3: Scale features (fit on train ONLY)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # ═══════════════════════════════════════════════════════
        # [ALERT] Phase 4: FEATURE SELECTION — reduces noise features
        #    Fit on TRAIN only, transform all sets
        #    This forces the model to learn from STRONG signals only
        # ═══════════════════════════════════════════════════════
        if feature_selection:
            from sklearn.feature_selection import SelectFromModel
            from sklearn.ensemble import RandomForestClassifier as _RFC
            
            n_features_before = X_train_scaled.shape[1]
            
            # Use a lightweight RF to identify important features
            selector_model = _RFC(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            )
            self.feature_selector = SelectFromModel(
                selector_model, threshold='median'
            )
            self.feature_selector.fit(X_train_scaled, y_train)
            
            X_train_scaled = self.feature_selector.transform(X_train_scaled)
            X_val_scaled = self.feature_selector.transform(X_val_scaled)
            X_test_scaled = self.feature_selector.transform(X_test_scaled)
            
            # Update feature names to match selected features
            mask = self.feature_selector.get_support()
            self.feature_names = [f for f, m in zip(self.feature_names, mask) if m]
            
            print(f"   [TEST] Feature Selection: {n_features_before} → {X_train_scaled.shape[1]} features (kept top {X_train_scaled.shape[1]})")
        
        # ═══════════════════════════════════════════════════════
        # [ALERT] Phase 5: SMOTE — synthetic oversampling (TRAIN only)
        #    Generates synthetic minority samples to balance classes
        #    WITHOUT using any real test/val data
        # ═══════════════════════════════════════════════════════
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                n_before = X_train_scaled.shape[0]
                sm = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_train.astype(int))) - 1))
                X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
                print(f"   ⚗️  SMOTE: Training samples {n_before} → {X_train_scaled.shape[0]}")
            except ImportError:
                print("   [WARN]  SMOTE skipped: install imbalanced-learn (`pip install imbalanced-learn`)")
            except Exception as e:
                print(f"   [WARN]  SMOTE failed: {e}")
        
        print(f"\n[OK] Data ready (LEAKAGE-FREE + ANTI-OVERFITTING pipeline)!")
        print(f"   Train set:      {X_train_scaled.shape} ({len(y_train)} samples)")
        print(f"   Validation set: {X_val_scaled.shape} ({len(y_val)} samples)")
        print(f"   Test set:       {X_test_scaled.shape} ({len(y_test)} samples)")
        print(f"   Features:       {len(self.feature_names)}")
        print("=" * 60)
        
        y_train_out = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_out = y_val.values if hasattr(y_val, 'values') else y_val
        y_test_out = y_test.values if hasattr(y_test, 'values') else y_test
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_out, y_val_out, y_test_out, self.feature_names


# Convenience function
def load_data(data_dir: str = "data/raw") -> Tuple:
    """Load, merge, and prepare kidney disease data (3-way split, leakage-free)."""
    loader = DataLoader(data_dir)
    return loader.load_and_prepare_data()


if __name__ == "__main__":
    loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test, features = loader.load_and_prepare_data()
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Features: {features}")
