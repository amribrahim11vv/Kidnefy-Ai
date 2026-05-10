"""
Text Extractor Module
Extract medical test values from lab result images using OCR.
"""

import os
import re
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# easyocr will be imported lazily to prevent Torch DLL Access Violation from crashing the server on startup
EASYOCR_AVAILABLE = False

try:
    import pytesseract
    # Configure tesseract path for Windows common installations
    if os.name == 'nt':
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Users\\' + os.getlogin() + r'\AppData\Local\Tesseract-OCR\tesseract.exe'
        ]
        for path in common_paths:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                break
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from .image_processor import ImageProcessor


@dataclass
class ExtractedValue:
    """Container for an extracted lab value."""
    test_name: str
    value: float
    unit: str
    reference_range: str
    is_abnormal: bool
    confidence: float


class LabImageExtractor:
    """
    Extract medical test values from lab result images.
    Uses EasyOCR as primary engine, Tesseract as fallback.
    """
    
    # Common medical test patterns
    TEST_PATTERNS = {
        'serum_creatinine': [
            r'serum\s*creatinine[:\s]*([0-9.]+)\s*(mg/dl|mg/dL)?',
            r'creatinine[:\s]*([0-9.]+)\s*(mg/dl|mg/dL)?',
            r's\.?\s*creatinine[:\s]*([0-9.]+)',
            r'cr[:\s]*([0-9.]+)\s*(mg/dl)?',
        ],
        'blood_urea': [
            r'blood\s*urea[:\s]*([0-9.]+)\s*(U/L|mg/dl)?',
            r'urea[:\s]*([0-9.]+)\s*(U/L|mg/dl)?',
            r'bun[:\s]*([0-9.]+)',
            r'blood\s*urea\s*nitrogen[:\s]*([0-9.]+)',
        ],
        'serum_urea': [
            r'serum\s*urea[:\s]*([0-9.]+)\s*(mg/dl)?',
        ],
        'albumin': [
            r'albumin[:\s]*([0-9.]+)\s*(g/dl|g/dL|ug/ml)?',
            r'alb[:\s]*([0-9.]+)',
            r'serum\s*albumin[:\s]*([0-9.]+)',
        ],
        'albumin_urine': [
            r'albumin\s*in\s*urine[:\s]*([0-9.]+)\s*(ug/mL)?',
            r'urine\s*albumin[:\s]*([0-9.]+)',
            r'microalbumin[:\s]*([0-9.]+)',
        ],
        'creatinine_urine': [
            r'creatinine\s*in\s*urine[:\s]*([0-9.]+)\s*(mg/dL)?',
            r'urine\s*creatinine[:\s]*([0-9.]+)',
        ],
        'acr': [
            r'albumin\s*creatinine\s*ratio[:\s]*([0-9.]+)\s*(mg/g)?',
            r'albumin[/]creatinine\s*ratio[:\s]*([0-9.]+)',
            r'acr[:\s]*([0-9.]+)',
            r'a/c\s*ratio[:\s]*([0-9.]+)',
            r'uacr[:\s]*([0-9.]+)',
        ],
        'egfr': [
            r'egfr[:\s]*([0-9.]+)',
            r'gfr[:\s]*([0-9.]+)',
            r'estimated\s*gfr[:\s]*([0-9.]+)',
            r'e\.?g\.?f\.?r[:\s]*([0-9.]+)',
        ],
        'hemoglobin': [
            r'hemoglobin[:\s]*([0-9.]+)\s*(g/dl)?',
            r'haemoglobin[:\s]*([0-9.]+)\s*(g/dl)?',
            r'hgb[:\s]*([0-9.]+)',
            r'hb[:\s]*([0-9.]+)',
        ],
        'sodium': [
            r'sodium[:\s]*([0-9.]+)\s*(mmol/L|mEq/L)?',
            r'na\+?[:\s]*([0-9.]+)',
            r'serum\s*sodium[:\s]*([0-9.]+)',
        ],
        'potassium': [
            r'potassium[:\s]*([0-9.]+)\s*(mmol/L|mEq/L)?',
            r'k\+?[:\s]*([0-9.]+)',
            r'serum\s*potassium[:\s]*([0-9.]+)',
        ],
        'calcium': [
            r'calcium[:\s]*([0-9.]+)\s*(mg/dl)?',
            r'serum\s*calcium[:\s]*([0-9.]+)',
            r'ca\+?\+?[:\s]*([0-9.]+)',
            r'total\s*calcium[:\s]*([0-9.]+)',
            r'ionized\s*calcium[:\s]*([0-9.]+)',
        ],
        'uric_acid': [
            r'uric\s*acid[:\s]*([0-9.]+)\s*(mg/dl)?',
            r'serum\s*uric\s*acid[:\s]*([0-9.]+)',
            r'urate[:\s]*([0-9.]+)',
        ],
        'blood_pressure_systolic': [
            r'bp[:\s]*([0-9]+)/([0-9]+)',
            r'blood\s*pressure[:\s]*([0-9]+)/([0-9]+)',
        ],
        'glucose': [
            r'glucose[:\s]*([0-9.]+)\s*(mg/dl)?',
            r'blood\s*sugar[:\s]*([0-9.]+)',
            r'fbs[:\s]*([0-9.]+)',
            r'fasting\s*blood\s*sugar[:\s]*([0-9.]+)',
            r'random\s*blood\s*sugar[:\s]*([0-9.]+)',
            r'rbs[:\s]*([0-9.]+)',
        ],
        # ===== Additional Kidney-Related Biomarkers =====
        'phosphorus': [
            r'phosphorus[:\s]*([0-9.]+)\s*(mg/dl)?',
            r'phosphate[:\s]*([0-9.]+)\s*(mg/dl)?',
            r'serum\s*phosphorus[:\s]*([0-9.]+)',
            r'po4[:\s]*([0-9.]+)',
        ],
        'bicarbonate': [
            r'bicarbonate[:\s]*([0-9.]+)\s*(mmol/L|mEq/L)?',
            r'hco3[:\s]*([0-9.]+)',
            r'co2[:\s]*([0-9.]+)\s*(mmol/L)?',
            r'total\s*co2[:\s]*([0-9.]+)',
        ],
        'chloride': [
            r'chloride[:\s]*([0-9.]+)\s*(mmol/L|mEq/L)?',
            r'cl[:\s]*([0-9.]+)\s*(mmol/L)?',
            r'serum\s*chloride[:\s]*([0-9.]+)',
        ],
        'total_protein': [
            r'total\s*protein[:\s]*([0-9.]+)\s*(g/dl)?',
            r'serum\s*protein[:\s]*([0-9.]+)',
            r't\.?\s*protein[:\s]*([0-9.]+)',
        ],
        'hba1c': [
            r'hba1c[:\s]*([0-9.]+)\s*(%)?',
            r'glycated\s*hemoglobin[:\s]*([0-9.]+)',
            r'a1c[:\s]*([0-9.]+)',
            r'hemoglobin\s*a1c[:\s]*([0-9.]+)',
        ],
        'wbc': [
            r'wbc[:\s]*([0-9.]+)\s*(10\^?3|x10|/ul)?',
            r'white\s*blood\s*cells?[:\s]*([0-9.]+)',
            r'leucocytes?[:\s]*([0-9.]+)',
        ],
        'rbc': [
            r'rbc[:\s]*([0-9.]+)\s*(10\^?6|x10|million)?',
            r'red\s*blood\s*cells?[:\s]*([0-9.]+)',
            r'erythrocytes?[:\s]*([0-9.]+)',
        ],
        'platelets': [
            r'platelets?[:\s]*([0-9.]+)\s*(10\^?3|x10|/ul)?',
            r'plt[:\s]*([0-9.]+)',
            r'platelet\s*count[:\s]*([0-9.]+)',
        ],
        'pcv': [
            r'pcv[:\s]*([0-9.]+)\s*(%)?',
            r'hct[:\s]*([0-9.]+)',
            r'hematocrit[:\s]*([0-9.]+)',
            r'packed\s*cell\s*volume[:\s]*([0-9.]+)',
        ],
        'urine_ph': [
            r'urine\s*ph[:\s]*([0-9.]+)',
            r'ph\s*\(urine\)[:\s]*([0-9.]+)',
            r'u\.?\s*ph[:\s]*([0-9.]+)',
        ],
        'specific_gravity': [
            r'specific\s*gravity[:\s]*([0-9.]+)',
            r'sp\.?\s*gr\.?[:\s]*([0-9.]+)',
            r'sg[:\s]*([0-9.]+)',
        ],
        'protein_urine': [
            r'protein\s*\(urine\)[:\s]*([0-9.]+)',
            r'urine\s*protein[:\s]*([0-9.]+)',
            r'proteinuria[:\s]*([0-9.]+)',
        ],
        # ── Arabic keywords (Egyptian lab reports) ──────────────────────
        # Arabic labels followed by a number on the same or next token
        'serum_creatinine': [
            r'serum\s*creatinine[:\s]*([0-9.]+)\s*(mg/dl|mg/dL)?',
            r'creatinine[:\s]*([0-9.]+)\s*(mg/dl|mg/dL)?',
            r's\.?\s*creatinine[:\s]*([0-9.]+)',
            r'cr[:\s]*([0-9.]+)\s*(mg/dl)?',
            r'\u0643\u0631\u064a\u0627\u062a\u064a\u0646\u064a\u0646[^0-9]*([0-9.]+)',  # كرياتينين
            r'\u0633\u064a\u0631\u0645[^0-9]*\u0643\u0631[^0-9]*([0-9.]+)',
        ],
    }
    
    # Reference ranges for common tests
    REFERENCE_RANGES = {
        'serum_creatinine': {'min': 0.5, 'max': 1.5, 'unit': 'mg/dL'},
        'blood_urea': {'min': 10, 'max': 50, 'unit': 'U/L'},
        'serum_urea': {'min': 10, 'max': 50, 'unit': 'mg/dL'},
        'albumin': {'min': 3.5, 'max': 5.0, 'unit': 'g/dL'},
        'albumin_urine': {'min': 0, 'max': 20, 'unit': 'ug/mL'},
        'creatinine_urine': {'min': 20, 'max': 320, 'unit': 'mg/dL'},
        'acr': {'min': 0, 'max': 30, 'unit': 'mg/g'},
        'egfr': {'min': 90, 'max': 120, 'unit': 'mL/min/1.73m²'},
        'hemoglobin': {'min': 12, 'max': 17, 'unit': 'g/dL'},
        'sodium': {'min': 136, 'max': 145, 'unit': 'mmol/L'},
        'potassium': {'min': 3.5, 'max': 5.0, 'unit': 'mmol/L'},
        'calcium': {'min': 8.8, 'max': 10.2, 'unit': 'mg/dL'},
        'uric_acid': {'min': 3.4, 'max': 7.0, 'unit': 'mg/dL'},
        'glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL'},
        'blood_pressure_systolic': {'min': 90, 'max': 140, 'unit': 'mmHg'},
        'blood_pressure_diastolic': {'min': 60, 'max': 90, 'unit': 'mmHg'},
        # New ranges
        'phosphorus': {'min': 2.5, 'max': 4.5, 'unit': 'mg/dL'},
        'bicarbonate': {'min': 22, 'max': 29, 'unit': 'mmol/L'},
        'chloride': {'min': 96, 'max': 106, 'unit': 'mmol/L'},
        'total_protein': {'min': 6.0, 'max': 8.3, 'unit': 'g/dL'},
        'hba1c': {'min': 4.0, 'max': 5.6, 'unit': '%'},
        'wbc': {'min': 4.5, 'max': 11.0, 'unit': '10^3/uL'},
        'rbc': {'min': 4.5, 'max': 5.5, 'unit': '10^6/uL'},
        'platelets': {'min': 150, 'max': 400, 'unit': '10^3/uL'},
        'pcv': {'min': 36, 'max': 48, 'unit': '%'},
        'urine_ph': {'min': 4.5, 'max': 8.0, 'unit': ''},
        'specific_gravity': {'min': 1.005, 'max': 1.030, 'unit': ''},
        'protein_urine': {'min': 0, 'max': 15, 'unit': 'mg/dL'},
    }
    
    def __init__(self, languages: List[str] = None):
        """
        Initialize the extractor.
        
        Args:
            languages: List of language codes for OCR (default: ['en'])
        """
        if languages is None:
            languages = ['en']  # safe runtime default avoids mutable-default-argument bug
        self.languages = languages
        self.image_processor = ImageProcessor()
        
        # Initialize OCR engine
        self.reader = None
        self.ocr_engine = None
        
        # Try to load EasyOCR lazily to avoid Torch startup crashes
        global EASYOCR_AVAILABLE
        try:
            import easyocr
            EASYOCR_AVAILABLE = True
            self.reader = easyocr.Reader(languages, gpu=False)
            self.ocr_engine = 'easyocr'
            print("Using EasyOCR engine")
        except Exception as e:
            print(f"EasyOCR initialization failed: {e}")
            EASYOCR_AVAILABLE = False
        
        if self.reader is None and PYTESSERACT_AVAILABLE:
            self.ocr_engine = 'tesseract'
            print("Using Tesseract OCR engine")
        
        if self.ocr_engine is None:
            print("Warning: No OCR engine available. Text extraction will fail.")
    
    def extract_text(
        self,
        image: Union[np.ndarray, str, Path, bytes],
        preprocess: bool = True
    ) -> Tuple[str, List[Dict]]:
        """
        Extract text from image using OCR.
        
        Args:
            image: Input image
            preprocess: Apply preprocessing
            
        Returns:
            Tuple of (full_text, list of detected regions with text and confidence)
        """
        # Preprocess image
        if preprocess:
            processed = self.image_processor.preprocess_for_ocr(image)
        else:
            if isinstance(image, (str, Path, bytes)):
                processed = self.image_processor.load_image(image)
            else:
                processed = image
        
        regions = []
        
        if self.ocr_engine == 'easyocr':
            results = self.reader.readtext(processed)
            for bbox, text, conf in results:
                regions.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': bbox
                })
            full_text = ' '.join([r['text'] for r in regions])
            
        else:  # Tesseract
            try:
                # Get detailed output
                data = pytesseract.image_to_data(
                    processed, 
                    output_type=pytesseract.Output.DICT
                )
                
                for i, text in enumerate(data['text']):
                    if text.strip():
                        conf = int(data['conf'][i]) / 100 if data['conf'][i] != -1 else 0
                        regions.append({
                            'text': text,
                            'confidence': conf,
                            'bbox': [
                                data['left'][i],
                                data['top'][i],
                                data['width'][i],
                                data['height'][i]
                            ]
                        })
                
                full_text = pytesseract.image_to_string(processed)
            except pytesseract.pytesseract.TesseractNotFoundError:
                raise RuntimeError("Tesseract OCR is not installed or not in PATH. Please install Tesseract-OCR to use this feature.")
            except Exception as e:
                raise RuntimeError(f"Tesseract OCR failed: {str(e)}")
        
        return full_text, regions
    
    # ── OCR error normalization map ────────────────────────────────────────
    _OCR_FIXES = [
        # Common character-level confusions
        (r'creotinine',  'creatinine'),
        (r'creetinine',  'creatinine'),
        (r'cre[ao]tine', 'creatinine'),
        (r'haemog[lo]+bin', 'hemoglobin'),
        (r'haemoglobin',  'hemoglobin'),
        (r'urea\s*nitr',  'blood urea nitrogen'),
        (r'bicarbonote',  'bicarbonate'),
        (r'calcuim',      'calcium'),
        (r'soduim',       'sodium'),
        (r'pottasium',    'potassium'),
        (r'potassim',     'potassium'),
        (r'chlioride',    'chloride'),
        (r'glucoze',      'glucose'),
        (r'g[1l]ucose',   'glucose'),
        # Zero/O and 1/l confusions in numbers
        (r'(?<=[0-9])[oO](?=[0-9])', '0'),
        (r'(?<=[0-9])[lI](?=[0-9])', '1'),
    ]

    def _normalize_text(self, text: str) -> str:
        """Fix common OCR mis-reads before regex extraction."""
        for pattern, replacement in self._OCR_FIXES:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def parse_values(self, text: str) -> Dict[str, ExtractedValue]:
        """
        Parse extracted text to find medical test values.

        Args:
            text: OCR extracted text

        Returns:
            Dictionary of detected test values
        """
        text = self._normalize_text(text)  # fix OCR errors first
        results = {}
        text_lower = text.lower()
        
        for test_name, patterns in self.TEST_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    try:
                        # Special handling for blood pressure (2-group capture: systolic/diastolic)
                        if test_name == 'blood_pressure_systolic' and len(match.groups()) >= 2:
                            systolic = float(match.group(1))
                            diastolic = float(match.group(2))
                            
                            # Store systolic
                            ref_sys = self.REFERENCE_RANGES.get('blood_pressure_systolic', {})
                            results['blood_pressure_systolic'] = ExtractedValue(
                                test_name='blood_pressure_systolic',
                                value=systolic,
                                unit='mmHg',
                                reference_range=f"{ref_sys.get('min', 90)} - {ref_sys.get('max', 140)}",
                                is_abnormal=systolic < ref_sys.get('min', 90) or systolic > ref_sys.get('max', 140),
                                confidence=0.8
                            )
                            
                            # Store diastolic
                            ref_dia = self.REFERENCE_RANGES.get('blood_pressure_diastolic', {})
                            results['blood_pressure_diastolic'] = ExtractedValue(
                                test_name='blood_pressure_diastolic',
                                value=diastolic,
                                unit='mmHg',
                                reference_range=f"{ref_dia.get('min', 60)} - {ref_dia.get('max', 90)}",
                                is_abnormal=diastolic < ref_dia.get('min', 60) or diastolic > ref_dia.get('max', 90),
                                confidence=0.8
                            )
                            break
                        
                        value = float(match.group(1))
                        unit = match.group(2) if len(match.groups()) > 1 and match.group(2) else ''
                        
                        # Get reference range
                        ref = self.REFERENCE_RANGES.get(test_name, {})
                        ref_min = ref.get('min', 0)
                        ref_max = ref.get('max', float('inf'))
                        ref_unit = ref.get('unit', '')
                        
                        # Check if abnormal
                        is_abnormal = value < ref_min or value > ref_max
                        
                        results[test_name] = ExtractedValue(
                            test_name=test_name,
                            value=value,
                            unit=unit or ref_unit,
                            reference_range=f"{ref_min} - {ref_max}",
                            is_abnormal=is_abnormal,
                            confidence=0.8  # Default confidence
                        )
                        break
                    except (ValueError, IndexError):
                        continue
        
        return results
    
    def extract_patient_info(self, text: str) -> Dict[str, Any]:
        """Extract patient information from text."""
        info = {}
        
        # Patient name
        name_match = re.search(r'patient\s*name[:\s]*([^\n]+)', text, re.IGNORECASE)
        if name_match:
            info['patient_name'] = name_match.group(1).strip()
        
        # Age
        age_match = re.search(r'age[:\s]*(\d+)\s*(years?)?', text, re.IGNORECASE)
        if age_match:
            info['age'] = int(age_match.group(1))
        
        # Sex
        sex_match = re.search(r'sex[:\s]*(male|female|m|f)', text, re.IGNORECASE)
        if sex_match:
            sex = sex_match.group(1).lower()
            info['sex'] = 'male' if sex in ['male', 'm'] else 'female'
        
        # Date
        date_match = re.search(r'date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE)
        if date_match:
            info['date'] = date_match.group(1)
        
        # Lab number
        lab_match = re.search(r'lab[.\s]*no[:\s]*(\d+)', text, re.IGNORECASE)
        if lab_match:
            info['lab_no'] = lab_match.group(1)
        
        return info
    
    def extract_all(
        self,
        image: Union[np.ndarray, str, Path, bytes]
    ) -> Dict[str, Any]:
        """
        Complete extraction pipeline.
        
        Args:
            image: Input lab result image
            
        Returns:
            Dictionary with all extracted data
        """
        # Extract text
        full_text, regions = self.extract_text(image)
        
        # Parse medical values
        test_values = self.parse_values(full_text)
        
        # Extract patient info
        patient_info = self.extract_patient_info(full_text)
        
        # Convert ExtractedValue objects to dict
        values_dict = {}
        for name, ev in test_values.items():
            values_dict[name] = {
                'value': ev.value,
                'unit': ev.unit,
                'reference_range': ev.reference_range,
                'is_abnormal': ev.is_abnormal,
                'confidence': ev.confidence
            }
        
        return {
            'raw_text': full_text,
            'patient_info': patient_info,
            'test_values': values_dict,
            'ocr_regions': regions,
            'ocr_engine': self.ocr_engine
        }
    
    def extract_for_prediction(
        self,
        image: Union[np.ndarray, str, Path, bytes]
    ) -> Dict[str, float]:
        """
        Extract values formatted for model prediction.
        
        Returns:
            Dictionary mapping feature names to values
        """
        data = self.extract_all(image)
        
        # Map to model features
        feature_mapping = {
            'serum_creatinine': 'sc',
            'blood_urea': 'bu',
            'serum_urea': 'bu',
            'hemoglobin': 'hemo',
            'sodium': 'sod',
            'potassium': 'pot',
            'albumin': 'al',
            'glucose': 'bgr'
        }
        
        features = {}
        for test_name, feature_name in feature_mapping.items():
            if test_name in data['test_values']:
                features[feature_name] = data['test_values'][test_name]['value']
        
        # Extract age if available
        if 'age' in data['patient_info']:
            features['age'] = data['patient_info']['age']
        
        return features
    
    def get_kidney_markers(
        self,
        image: Union[np.ndarray, str, Path, bytes]
    ) -> Dict[str, Any]:
        """
        Extract key kidney function markers for staging.
        
        Returns:
            Dictionary with creatinine, ACR, eGFR, and patient info
        """
        data = self.extract_all(image)
        
        markers = {
            'creatinine': None,
            'acr': None,
            'egfr': None,
            'age': None,
            'sex': None,
        }
        
        if 'serum_creatinine' in data['test_values']:
            markers['creatinine'] = data['test_values']['serum_creatinine']['value']
        
        if 'acr' in data['test_values']:
            markers['acr'] = data['test_values']['acr']['value']
        
        if 'egfr' in data['test_values']:
            markers['egfr'] = data['test_values']['egfr']['value']
        
        if 'age' in data['patient_info']:
            markers['age'] = data['patient_info']['age']
        
        if 'sex' in data['patient_info']:
            markers['sex'] = data['patient_info']['sex']
        
        return markers


if __name__ == "__main__":
    # Test the extractor
    extractor = LabImageExtractor()
    
    # Test with sample text
    sample_text = """
    Patient name: Mohamed Ahmed
    Age: 70 Years
    Sex: Male
    Date: 21/05/2025
    Lab. No: 1806
    
    Kidney Function Test
    Serum Creatinine: 2.3 mg/dl   (0.5 - 1.5)
    Serum Urea: 61 mg/dl          (10 - 50)
    
    Albumin/creatinine Ratio
    Creatinine in Urine: 32.40 mg/dL
    Albumin in Urine: 14.4 ug/mL
    Albumin Creatinine Ratio: 44.44 mg/g
    """
    
    results = extractor.parse_values(sample_text)
    print("Extracted values:")
    for name, value in results.items():
        print(f"  {name}: {value.value} {value.unit} (Abnormal: {value.is_abnormal})")
    
    patient = extractor.extract_patient_info(sample_text)
    print(f"\nPatient info: {patient}")
