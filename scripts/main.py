"""
Kidney Disease Prediction System
Main entry point for training and using the AI model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Import modules
from src.preprocessing import DataLoader, FeatureEngineer, calculate_egfr
from src.models import EnsembleModel, MLModels, DeepLearningModel
# from src.ocr import LabImageExtractor, ImageProcessor
from src.staging import GFRCalculator, RiskAssessor
from src.reports import PDFReportGenerator, PatientInfo, TestResult
from src.explainability import SHAPExplainer
from src.monitoring import LongitudinalMonitor
from config import CKD_FEATURE_ORDER, CKD_FEATURE_DEFAULTS


class KidneyDiseasePredictionSystem:
    """
    Complete system for kidney disease prediction.
    Combines ML/DL models, OCR, staging, and report generation.
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.ensemble_model = EnsembleModel(str(self.model_dir))
        self.gfr_calculator = GFRCalculator()
        self.risk_assessor = RiskAssessor()
        self.report_generator = PDFReportGenerator()
        
        # XAI - SHAP Explainer
        self.shap_explainer = SHAPExplainer()
        
        # Longitudinal Monitoring
        self.longitudinal_monitor = LongitudinalMonitor()
        
        # Store feature names and training data reference for SHAP
        self._feature_names = None
        self._X_train_sample = None
        
        # OCR will be initialized on demand
        self._ocr_extractor = None
        
        self.is_trained = False
    
    @property
    def ocr_extractor(self):
        """Lazy initialization of OCR extractor."""
        if self._ocr_extractor is None:
            from src.ocr import LabImageExtractor
            self._ocr_extractor = LabImageExtractor()
        return self._ocr_extractor
    
    def train(self, epochs: int = 50, ckd_only: bool = False):
        """Train all models on the dataset.
        
        Args:
            epochs: Training epochs for Deep Learning model
            ckd_only: If True, train on real UCI CKD data only (83-90% acc).
                      If False, merge all datasets including synthetic (95-99% acc).
        """
        print("=" * 60)
        print("Kidney Disease Prediction System - Training")
        print(f"   Mode: {'UCI CKD only (realistic)' if ckd_only else 'All datasets (combined)'}")
        print("=" * 60)
        
        # Load and prepare data (3-way split: Train/Val/Test)
        print("\n Loading and preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
            self.data_loader.load_and_prepare_data(ckd_only=ckd_only)

        print(f"   Training samples:   {X_train.shape[0]}")
        print(f"   Validation samples: {X_val.shape[0]}")
        print(f"   Test samples:       {X_test.shape[0]}")
        print(f"   Features:           {len(feature_names)}")
        
        # Train ensemble (Val set for DL Early Stopping, Test for final eval only)
        print("\n Training models...")
        metrics = self.ensemble_model.train(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            dl_epochs=epochs
        )
        
        # Initialize SHAP Explainer with XGBoost model
        print("\n Initializing SHAP Explainer...")
        try:
            xgb_model = self.ensemble_model.ml_models.models.get('XGBoost')
            if xgb_model:
                self.shap_explainer.fit(xgb_model, X_train, model_type="tree")
            self._feature_names = feature_names
            self._X_train_sample = X_train[:200]  # Store sample for SHAP background
        except Exception as e:
            print(f"   [WARN] SHAP initialization failed: {e}")
        
        # Save models
        print("\n Saving models...")
        self.ensemble_model.feature_names = feature_names  # Persist for inference
        self.ensemble_model.ml_models.feature_names = feature_names
        self.ensemble_model.scaler = self.data_loader.scaler
        self.ensemble_model.save()
        
        self.is_trained = True
        
        print("\n[OK] Training complete!")
        return metrics
    
    def predict_from_features(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Make prediction from feature dictionary.
        
        Args:
            features: Dictionary mapping feature names to values
                Required: 'sc' (creatinine), optional: age, egfr, acr, etc.
        
        Returns:
            Complete prediction result
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Extract key values
        creatinine = features.get('sc', features.get('creatinine', 1.0))
        age = features.get('age', 50)
        is_female = features.get('is_female', False)
        acr = features.get('acr', None)
        
        # Calculate eGFR
        egfr = features.get('egfr')
        if egfr is None:
            egfr = calculate_egfr(creatinine, age, is_female)
        
        # Add calculated eGFR to features if not present
        if 'egfr' not in features:
            features['egfr'] = egfr

        # Prepare feature vector matching the training features
        expected_features = CKD_FEATURE_ORDER
        defaults = CKD_FEATURE_DEFAULTS
        
        feature_dict = {}
        for f in expected_features:
            # Handle potential aliases in input features dict
            val = features.get(f)
            if val is None and f == "uacr":
                val = features.get("acr")
            if val is None:
                # Aliases mapping
                aliases = {
                    'bp_dia': 'blood_pressure_diastolic',
                    'serum_albumin': 'albumin', # if user sends 'albumin' for serum
                    'hba1c': 'HbA1c'
                }
                for alias in aliases:
                    if f == alias and aliases[alias] in features:
                        val = features[aliases[alias]]
                        break
            
            # Use default if still None
            if val is None:
                val = defaults.get(f, 0)
            
            feature_dict[f] = [val] # Wrap in list for DataFrame

        # Convert to DataFrame
        df_features = pd.DataFrame(feature_dict)
        
        # Apply exactly the same Medical Feature Categorization as training
        df_features = self.feature_engineer.create_categorical_bins(df_features)
        
        # Ensure the feature space perfectly matches the trained model
        if hasattr(self.ensemble_model.ml_models, 'feature_names') and self.ensemble_model.ml_models.feature_names:
            trained_features = self.ensemble_model.ml_models.feature_names
            # Add missing dummy columns with 0
            for col in trained_features:
                if col not in df_features.columns:
                    df_features[col] = 0
            # Keep only the exact columns in the exact order
            df_features = df_features[trained_features]
            
        df_features = df_features.astype(float)
        if getattr(self.ensemble_model, "scaler", None) is not None:
            try:
                feature_vector = self.ensemble_model.scaler.transform(df_features)
            except Exception as e:
                print(f"[WARN] Scaler transformation failed: {e}. Falling back to raw features.")
                feature_vector = df_features.values
        else:
            feature_vector = df_features.values
        
        # Get ensemble prediction
        pred, confidence, details = self.ensemble_model.predict_with_confidence(feature_vector)
        probability = details['ensemble_proba'][0]
        
        # Get complete assessment with enhanced biomarkers
        other_values = {
            'hba1c': features.get('hba1c', 5.5),
            'uric_acid': features.get('uric_acid', 5.0),
            'bmi': features.get('bmi', 25.0),
            'smoking': features.get('smoking', 0),
            'diabetes_duration': features.get('diabetes_duration', 0),
        }
        assessment = self.risk_assessor.complete_assessment(
            ckd_probability=probability,
            creatinine=creatinine,
            egfr=egfr,
            acr=acr,
            age=age,
            is_female=is_female,
            other_values=other_values
        )
        
        result = {
            'prediction': bool(pred[0]),
            'probability': float(probability),
            'confidence': float(confidence[0]),
            'egfr': egfr,
            'gfr_stage': assessment.gfr_stage.value,
            'albuminuria_category': assessment.albuminuria_category.value if assessment.albuminuria_category else None,
            'risk_level': assessment.risk_level.value,
            'progression_risk': assessment.progression_risk.risk_percentage,
            'enhanced_risk_score': assessment.enhanced_risk_score,
            'recommendations': assessment.recommendations,
            'alerts': assessment.alerts
        }
        
        # Add SHAP explanation if available
        if self.shap_explainer.explainer is not None and self._feature_names:
            try:
                explanation = self.shap_explainer.explain_prediction(
                    feature_vector, self._feature_names
                )
                result['xai_explanation'] = {
                    'top_risk_factors': explanation.get('top_risk_factors', [])[:5],
                    'top_protective_factors': explanation.get('top_protective_factors', [])[:5],
                    'explanation_text': explanation.get('explanation_text', '')
                }
            except Exception as e:
                result['xai_explanation'] = {'error': str(e)}
        
        return result
    
    def add_patient_measurement(
        self,
        patient_id: str,
        date: str,
        egfr: float,
        creatinine: float = None,
        uacr: float = None,
        hba1c: float = None,
    ) -> Dict[str, Any]:
        """
        Add a longitudinal measurement for a patient.
        
        Args:
            patient_id: Unique patient identifier
            date: Date (YYYY-MM-DD)
            egfr: eGFR value
            creatinine: Serum creatinine
            uacr: UACR value
            hba1c: HbA1c percentage
        """
        return self.longitudinal_monitor.add_measurement(
            patient_id=patient_id,
            date=date,
            egfr=egfr,
            creatinine=creatinine,
            uacr=uacr,
            hba1c=hba1c
        )
    
    def get_patient_trend(self, patient_id: str) -> Dict[str, Any]:
        """
        Get longitudinal trend analysis for a patient.
        Includes fast progressor detection.
        """
        from dataclasses import asdict
        trend = self.longitudinal_monitor.calculate_trend(patient_id)
        return asdict(trend)
    
    def predict_from_image(
        self,
        image_path: str,
        patient_age: int = 50,
        patient_is_female: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction from lab result image.
        
        Args:
            image_path: Path to lab result image
            patient_age: Patient age (Optional, will try to extract from image)
            patient_is_female: Patient sex
            
        Returns:
            Complete prediction result with extracted values
        """
        print(f" Processing image: {image_path}")
        
        # Extract data from image
        extraction = self.ocr_extractor.extract_all(image_path)
        
        # Get patient info
        patient_info = extraction.get('patient_info', {})
        age = patient_info.get('age', patient_age)
        is_female = patient_info.get('sex', 'male').lower() == 'female' if patient_info.get('sex') else patient_is_female
        
        # Get test values
        test_values = extraction.get('test_values', {})
        
        # Extract key markers
        creatinine = test_values.get('serum_creatinine', {}).get('value')
        acr = test_values.get('acr', {}).get('value')
        egfr = test_values.get('egfr', {}).get('value')
        
        if creatinine is None:
            raise ValueError("Could not extract creatinine from image")
        
        # Calculate eGFR if not extracted
        if egfr is None:
            egfr = calculate_egfr(creatinine, age, is_female)
        
        # Get staging
        staging = self.gfr_calculator.calculate_stage(
            creatinine=creatinine,
            egfr=egfr,
            acr=acr,
            age=age,
            is_female=is_female
        )
        
        return {
            'extracted_values': test_values,
            'patient_info': patient_info,
            'egfr': egfr,
            'gfr_stage': staging.gfr_stage.value,
            'albuminuria_category': staging.albuminuria_category.value if staging.albuminuria_category else None,
            'risk_level': staging.risk_level.value,
            'description': staging.description,
            'recommendations': staging.recommendations,
            'raw_text': extraction.get('raw_text', '')
        }
    
    def generate_report(
        self,
        result: Dict[str, Any],
        patient_name: str = "Patient",
        output_path: str = None
    ) -> str:
        """
        Generate PDF report from prediction result.
        
        Returns:
            Path to generated PDF
        """
        # Create patient info
        patient = PatientInfo(
            name=patient_name,
            age=result.get('patient_info', {}).get('age', 50),
            sex=result.get('patient_info', {}).get('sex', 'Unknown'),
            date=result.get('patient_info', {}).get('date', 'Today'),
            lab_no=result.get('patient_info', {}).get('lab_no', '')
        )
        
        # Create lab results
        lab_results = []
        extracted = result.get('extracted_values', {})
        for name, data in extracted.items():
            if isinstance(data, dict) and 'value' in data:
                lab_results.append(TestResult(
                    name=name.replace('_', ' ').title(),
                    value=data['value'],
                    unit=data.get('unit', ''),
                    reference_range=data.get('reference_range', ''),
                    is_abnormal=data.get('is_abnormal', False)
                ))
        
        # Generate report
        filepath = self.report_generator.generate_report(
            patient=patient,
            prediction=result.get('prediction', True),
            probability=result.get('probability', 0.5),
            risk_level=result.get('risk_level', 'Unknown'),
            gfr_stage=result.get('gfr_stage', 'Unknown'),
            egfr=result.get('egfr', 0),
            alb_category=result.get('albuminuria_category'),
            acr=extracted.get('acr', {}).get('value'),
            lab_results=lab_results,
            recommendations=result.get('recommendations', []),
            alerts=result.get('alerts', []),
            filename=output_path
        )
        
        return filepath


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Kidney Disease Prediction System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    train_parser.add_argument(
        '--ckd-only',
        action='store_true',
        default=False,
        help=(
            'Train on real UCI CKD dataset only (400 records). '
            'Expected accuracy: 83-90%%. Omit this flag to merge all '
            'datasets including synthetic data (expected accuracy: 95-99%%).'
        )
    )

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make prediction')
    predict_parser.add_argument('--image', type=str, help='Path to lab image')
    predict_parser.add_argument('--creatinine', type=float, help='Serum creatinine value')
    predict_parser.add_argument('--age', type=int, default=50, help='Patient age')
    predict_parser.add_argument('--acr', type=float, help='ACR value')
    
    # Stage command
    stage_parser = subparsers.add_parser('stage', help='Calculate kidney stage')
    stage_parser.add_argument('--creatinine', type=float, required=True)
    stage_parser.add_argument('--age', type=int, required=True)
    stage_parser.add_argument('--acr', type=float)
    stage_parser.add_argument('--female', action='store_true')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        system = KidneyDiseasePredictionSystem()
        system.train(epochs=args.epochs, ckd_only=args.ckd_only)

    elif args.command == 'predict':
        system = KidneyDiseasePredictionSystem()
        # Load trained models
        try:
            system.ensemble_model.load()
            system.is_trained = system.ensemble_model.is_trained
        except Exception as e:
            print(f"Error: Could not load trained models: {e}")
            print("Please train the model first: python main.py train")
            return
        
        if args.image:
            result = system.predict_from_image(args.image, args.age)
            print("\nPrediction Result:")
            print(f"  GFR Stage: {result['gfr_stage']}")
            print(f"  Risk Level: {result['risk_level']}")
            
    elif args.command == 'stage':
        calculator = GFRCalculator()
        result = calculator.calculate_stage(
            creatinine=args.creatinine,
            acr=args.acr,
            age=args.age,
            is_female=args.female
        )
        print(calculator.format_result(result))
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
