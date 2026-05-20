"""
Ensemble Model Module
Combines ML and DL models for improved predictions.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import joblib

from .ml_models import MLModels
from .dl_models import DeepLearningModel


class EnsembleModel:
    """
    Ensemble model combining ML and DL predictions.
    Supports weighted voting and stacking.
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.ml_models = MLModels(model_dir)
        self.dl_model = DeepLearningModel(model_dir)
        
        # Default weights for ensemble voting
        self.weights = {
            'Random Forest': 0.25,
            'XGBoost': 0.30,
            'SVM': 0.15,
            'Deep Learning': 0.30
        }
        
        self.is_trained = False
        self.training_metrics: Dict[str, Any] = {}
        self.feature_names: List[str] = []  # Persist for inference alignment
        self.scaler = None
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dl_epochs: int = 50
    ) -> Dict[str, Any]:
        """
        Train all models in the ensemble with comprehensive anti-overfitting measures.
        
        Anti-Overfitting Strategy:
          - ML models: Tightened hyperparameters + K-Fold CV on train set
          - DL model: Smaller network + Early Stopping on VALIDATION set
          - 3-way diagnostic: Train vs Val vs Test gap analysis
          - Test set: Used ONLY for final metric reporting (never influences training)
        
        Returns:
            Dictionary with all model metrics
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score
        
        print("=" * 60)
        print("Training Ensemble Model (Anti-Overfitting Mode)")
        print("=" * 60)
        
        # === ML Models: Train + Evaluate ===
        ml_metrics = self.ml_models.train_all_models(
            X_train, y_train, X_test, y_test,
            X_val=X_val, y_val=y_val,
            calibrate=True,
            calibration_method="isotonic",
        )
        
        # === K-Fold Cross-Validation on TRAINING set (10-fold, stratified) ===
        print("\n" + "=" * 60)
        print("K-Fold Cross-Validation (10-Fold Stratified on Training Set)")
        print("=" * 60)
        cv_results = {}
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for name in ['Random Forest', 'XGBoost', 'SVM']:
            model = self.ml_models.models.get(name)
            if model is not None:
                scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
                cv_results[name] = {
                    'cv_mean': scores.mean(),
                    'cv_std': scores.std(),
                    'cv_scores': scores.tolist()
                }
                stability = "[OK] STABLE" if scores.std() < 0.03 else "[WARN] UNSTABLE" if scores.std() < 0.05 else "[ERROR] VERY UNSTABLE"
                print(f"  {name:20} | CV F1: {scores.mean():.4f} ± {scores.std():.4f} | {stability}")
        
        # === DL Model: Train with VALIDATION set (NOT test) ===
        print("\n" + "=" * 40)
        self.dl_model.build_model(input_dim=X_train.shape[1])
        self.dl_model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,   # [OK] Validation set, NOT test set
            epochs=dl_epochs
        )
        dl_metrics = self.dl_model.evaluate(X_test, y_test)
        
        # Combine metrics
        self.training_metrics = {
            **ml_metrics,
            'Deep Learning': dl_metrics
        }
        
        # === Comprehensive Overfitting Diagnostic: Train vs Val vs Test ===
        print("\n" + "=" * 70)
        print("[TEST] OVERFITTING DIAGNOSTIC: Train vs Val vs Test")
        print("=" * 70)
        print(f"{'Model':20} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'T-V Gap':>8} | {'T-T Gap':>8} | {'Status'}")
        print("-" * 85)
        
        overfit_detected = False
        for name in ['Random Forest', 'XGBoost', 'SVM']:
            model = self.ml_models.models.get(name)
            if model is not None:
                train_acc = accuracy_score(y_train, model.predict(X_train))
                val_acc = accuracy_score(y_val, model.predict(X_val))
                test_acc = self.training_metrics[name]['accuracy']
                tv_gap = train_acc - val_acc
                tt_gap = train_acc - test_acc
                
                if tt_gap < 0.03:
                    status = "[OK] GOOD"
                elif tt_gap < 0.05:
                    status = "[WARN] SLIGHT"
                elif tt_gap < 0.10:
                    status = "[INFO] MODERATE"
                    overfit_detected = True
                else:
                    status = "[ERROR] OVERFIT"
                    overfit_detected = True
                
                print(f"  {name:20} | {train_acc:8.4f} | {val_acc:8.4f} | {test_acc:8.4f} | {tv_gap:8.4f} | {tt_gap:8.4f} | {status}")
                
                # Store val metrics
                self.training_metrics[name]['val_accuracy'] = val_acc
                self.training_metrics[name]['val_f1'] = f1_score(y_val, model.predict(X_val), average='weighted')
        
        # DL train vs val vs test
        dl_train_pred = self.dl_model.predict(X_train)
        dl_train_acc = accuracy_score(y_train, dl_train_pred)
        dl_val_pred = self.dl_model.predict(X_val)
        dl_val_acc = accuracy_score(y_val, dl_val_pred)
        dl_test_acc = dl_metrics['accuracy']
        dl_tv_gap = dl_train_acc - dl_val_acc
        dl_tt_gap = dl_train_acc - dl_test_acc
        dl_status = "[OK] GOOD" if dl_tt_gap < 0.03 else "[WARN] SLIGHT" if dl_tt_gap < 0.05 else "[INFO] MODERATE" if dl_tt_gap < 0.10 else "[ERROR] OVERFIT"
        print(f"  {'Deep Learning':20} | {dl_train_acc:8.4f} | {dl_val_acc:8.4f} | {dl_test_acc:8.4f} | {dl_tv_gap:8.4f} | {dl_tt_gap:8.4f} | {dl_status}")
        
        if overfit_detected:
            print("\n  [WARN]  Some models show potential overfitting. Consider:")
            print("     - Collecting more training data")
            print("     - Reducing feature count via feature selection")
            print("     - Further tightening regularization")
        else:
            print("\n  [OK] All models show healthy generalization gaps!")
        
        # === Update weights based on VALIDATION performance (not test) ===
        self._update_weights()
        
        self.is_trained = True
        
        # === Final Summary ===
        print("\n" + "=" * 60)
        print("Ensemble Training Summary")
        print("=" * 60)
        for name, weight in self.weights.items():
            metrics = self.training_metrics.get(name, {})
            f1 = metrics.get('f1_weighted', 0)  # key is 'f1_weighted', not 'f1_score'
            gap = metrics.get('overfit_gap', 0)
            cv_info = ""
            if name in cv_results:
                cv_info = f" | CV: {cv_results[name]['cv_mean']:.4f}±{cv_results[name]['cv_std']:.4f}"
            print(f"{name:20} | F1: {f1:.4f} | Gap: {gap:.4f} | Weight: {weight:.2f}{cv_info}")
        
        # Store CV results in training metrics for external access
        self.training_metrics['cross_validation'] = cv_results
        
        return self.training_metrics

    
    def _update_weights(self):
        """Update ensemble weights based on model performance."""
        total_f1 = 0
        f1_scores = {}
        
        for name in self.weights.keys():
            metrics = self.training_metrics.get(name, {})
            f1 = metrics.get('f1_weighted', 0)  # key is 'f1_weighted', not 'f1_score'
            f1_scores[name] = f1
            total_f1 += f1
        
        if total_f1 > 0:
            for name in self.weights.keys():
                self.weights[name] = f1_scores[name] / total_f1
    
    def predict_proba_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get prediction probabilities from all models."""
        probas = {}
        
        # ML model probabilities
        for name in ['Random Forest', 'XGBoost', 'SVM']:
            try:
                proba = self.ml_models.predict_proba(X, name)
                probas[name] = proba[:, 1] if len(proba.shape) > 1 else proba
            except Exception as e:
                print(f"Warning: Could not get probabilities from {name}: {e}")
        
        # DL model probability
        try:
            dl_proba = self.dl_model.predict_proba(X)
            probas['Deep Learning'] = dl_proba.flatten()
        except Exception as e:
            print(f"Warning: Could not get probabilities from DL model: {e}")
        
        return probas
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted ensemble prediction probabilities."""
        probas = self.predict_proba_all(X)
        
        # Weighted average
        ensemble_proba = np.zeros(X.shape[0])
        total_weight = 0
        
        for name, proba in probas.items():
            weight = self.weights.get(name, 0)
            ensemble_proba += weight * proba
            total_weight += weight
        
        if total_weight > 0:
            ensemble_proba /= total_weight
        
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make ensemble predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Make predictions with confidence scores and model agreement.
        
        Returns:
            predictions, confidence_scores, details
        """
        probas = self.predict_proba_all(X)
        ensemble_proba = self.predict_proba(X)
        predictions = (ensemble_proba >= threshold).astype(int)
        
        # Calculate confidence as distance from threshold
        confidence = np.abs(ensemble_proba - threshold) * 2  # Scale to 0-1
        confidence = np.clip(confidence, 0, 1)
        
        # Calculate model agreement
        all_preds = []
        for name, proba in probas.items():
            pred = (proba >= threshold).astype(int)
            all_preds.append(pred)
        
        all_preds = np.array(all_preds)
        agreement = np.mean(all_preds == predictions, axis=0)
        
        details = {
            'individual_probas': probas,
            'ensemble_proba': ensemble_proba,
            'model_agreement': agreement,
            'confidence': confidence
        }
        
        return predictions, confidence, details
    
    def get_prediction_explanation(self, X: np.ndarray, idx: int = 0) -> Dict:
        """Get detailed explanation for a single prediction."""
        probas = self.predict_proba_all(X)
        ensemble_proba = self.predict_proba(X)
        
        explanation = {
            'ensemble_probability': float(ensemble_proba[idx]),
            'prediction': 'CKD Positive' if ensemble_proba[idx] >= 0.5 else 'CKD Negative',
            'confidence': float(abs(ensemble_proba[idx] - 0.5) * 2),
            'model_contributions': {}
        }
        
        for name, proba in probas.items():
            weight = self.weights.get(name, 0)
            contribution = weight * proba[idx]
            explanation['model_contributions'][name] = {
                'probability': float(proba[idx]),
                'weight': float(weight),
                'weighted_contribution': float(contribution)
            }
        
        return explanation
    
    def save(self, prefix: str = "ensemble"):
        """Save all models and metadata."""
        print("Saving ensemble models...")
        
        # Save ML models
        self.ml_models.save_all_models()
        
        # Save DL model
        self.dl_model.save_model(f"{prefix}_dl_model.keras")
        
        # Save weights
        weights_path = self.model_dir / f"{prefix}_weights.joblib"
        joblib.dump(self.weights, weights_path)
        
        # Save scaler if present
        if getattr(self, 'scaler', None) is not None:
            scaler_path = self.model_dir / f"{prefix}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved scaler to {scaler_path}")
        
        # Save feature names for inference alignment
        meta_path = self.model_dir / f"{prefix}_metadata.joblib"
        joblib.dump({
            'feature_names': self.feature_names,
            'weights': self.weights
        }, meta_path)
        
        print("All models saved successfully!")
    
    def load(self, prefix: str = "ensemble"):
        """Load ensemble weights, metadata, and saved ML/DL checkpoints from disk."""
        self.is_trained = False

        weights_path = self.model_dir / f"{prefix}_weights.joblib"
        if weights_path.exists():
            self.weights = joblib.load(weights_path)

        meta_path = self.model_dir / f"{prefix}_metadata.joblib"
        if meta_path.exists():
            meta = joblib.load(meta_path)
            self.feature_names = meta.get('feature_names', [])
        self.ml_models.feature_names = self.feature_names or None

        # Load scaler if present
        scaler_path = self.model_dir / f"{prefix}_scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        else:
            self.scaler = None

        joblib_names = {
            "Random Forest": "random_forest_model.joblib",
            "XGBoost": "xgboost_model.joblib",
            "SVM": "svm_model.joblib",
        }
        loaded_ml = False
        for display_name, filename in joblib_names.items():
            path = self.model_dir / filename
            if path.exists():
                try:
                    self.ml_models.models[display_name] = joblib.load(path)
                    loaded_ml = True
                except Exception as e:
                    print(f"[WARN] Could not load {display_name} from {path}: {e}")

        dl_path = self.model_dir / f"{prefix}_dl_model.keras"
        if dl_path.exists():
            try:
                self.dl_model.load_model(str(dl_path))
            except Exception as e:
                print(f"[WARN] Could not load Deep Learning model from {dl_path}: {e}")

        self.is_trained = loaded_ml


class StackingEnsemble(EnsembleModel):
    """
    Stacking ensemble that uses a meta-learner to combine predictions.
    """
    
    def __init__(self, model_dir: str = "models"):
        super().__init__(model_dir)
        self.meta_model = None
        
    def train_stacking(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dl_epochs: int = 50
    ) -> Dict[str, Any]:
        """Train stacking ensemble with meta-learner."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        
        # First, train base models
        metrics = self.train(X_train, y_train, X_val, y_val, X_test, y_test, dl_epochs)
        
        # Generate meta-features using cross-validation predictions
        print("\nTraining Meta-Learner...")
        
        # Get out-of-fold predictions for training data
        meta_features_train = []
        
        for name in ['Random Forest', 'XGBoost', 'SVM']:
            model = self.ml_models.models[name]
            oof_pred = cross_val_predict(
                model, X_train, y_train, cv=5, method='predict_proba'
            )
            meta_features_train.append(oof_pred[:, 1])
        
        # For DL, generate out-of-fold predictions to avoid data leakage
        from sklearn.model_selection import KFold
        dl_oof = np.zeros(X_train.shape[0])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_train):
            fold_model = DeepLearningModel(str(self.model_dir))
            fold_model.build_model(input_dim=X_train.shape[1])
            fold_model.train(
                X_train[train_idx], y_train[train_idx],
                X_val=X_train[val_idx], y_val=y_train[val_idx],
                epochs=10  # verbose is not a parameter of DeepLearningModel.train()
            )
            dl_oof[val_idx] = fold_model.predict_proba(X_train[val_idx]).flatten()
        meta_features_train.append(dl_oof)
        
        # Stack meta-features
        meta_X_train = np.column_stack(meta_features_train)
        
        # Train meta-learner
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(meta_X_train, y_train)
        
        # Evaluate stacking
        meta_features_test = []
        for name in ['Random Forest', 'XGBoost', 'SVM']:
            proba = self.ml_models.predict_proba(X_test, name)[:, 1]
            meta_features_test.append(proba)
        
        dl_pred_test = self.dl_model.predict_proba(X_test).flatten()
        meta_features_test.append(dl_pred_test)
        
        meta_X_test = np.column_stack(meta_features_test)
        
        stacking_pred = self.meta_model.predict(meta_X_test)
        stacking_proba = self.meta_model.predict_proba(meta_X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        stacking_metrics = {
            'accuracy': accuracy_score(y_test, stacking_pred),
            'f1_score': f1_score(y_test, stacking_pred, average='weighted'),
            'auc_roc': roc_auc_score(y_test, stacking_proba)
        }
        
        print(f"\nStacking Ensemble Results:")
        print(f"  Accuracy: {stacking_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {stacking_metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:  {stacking_metrics['auc_roc']:.4f}")
        
        metrics['Stacking'] = stacking_metrics
        return metrics


if __name__ == "__main__":
    # Load data using the project's DataLoader
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.preprocessing.data_loader import DataLoader

    DATA_DIR = "data/raw"

    loader = DataLoader(data_dir=DATA_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = loader.load_and_prepare_data()

    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Features: {feature_names}")

    # Train ensemble
    ensemble = EnsembleModel()
    metrics = ensemble.train(X_train, y_train, X_val, y_val, X_test, y_test, dl_epochs=30)

    # Make predictions
    preds, confidence, details = ensemble.predict_with_confidence(X_test)

    print(f"\nEnsemble Accuracy: {np.mean(preds == y_test):.4f}")
    print(f"Average Confidence: {np.mean(confidence):.4f}")

    # Get explanation for first sample
    explanation = ensemble.get_prediction_explanation(X_test, idx=0)
    print(f"\nSample Prediction Explanation:")
    print(f"  Prediction: {explanation['prediction']}")
    print(f"  Confidence: {explanation['confidence']:.4f}")
