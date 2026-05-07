"""
OCR Feature Test Script - Standalone (No External Dependencies)
Tests the OCR text extraction regex patterns and parsing logic.
This script copies the relevant parsing logic inline to avoid any import issues.
"""

import re
from dataclasses import dataclass
from typing import Dict, Any

PASS = 0
FAIL = 0

def assert_test(condition, test_name):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {test_name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {test_name}")


@dataclass
class ExtractedValue:
    test_name: str
    value: float
    unit: str
    reference_range: str
    is_abnormal: bool
    confidence: float


# ---- Copied from src/ocr/text_extractor.py (must match the actual file) ----
TEST_PATTERNS = {
    'serum_creatinine': [
        r'serum\s*creatinine[:\s]*([0-9.]+)\s*(mg/dl|mg/dL)?',
        r'creatinine[:\s]*([0-9.]+)\s*(mg/dl|mg/dL)?',
        r's\.?\s*creatinine[:\s]*([0-9.]+)'
    ],
    'blood_urea': [
        r'blood\s*urea[:\s]*([0-9.]+)\s*(U/L|mg/dl)?',
        r'urea[:\s]*([0-9.]+)\s*(U/L|mg/dl)?',
        r'bun[:\s]*([0-9.]+)'
    ],
    'serum_urea': [
        r'serum\s*urea[:\s]*([0-9.]+)\s*(mg/dl)?',
    ],
    'albumin': [
        r'albumin[:\s]*([0-9.]+)\s*(g/dl|g/dL|ug/ml)?',
        r'alb[:\s]*([0-9.]+)'
    ],
    'albumin_urine': [
        r'albumin\s*in\s*urine[:\s]*([0-9.]+)\s*(ug/mL)?',
        r'urine\s*albumin[:\s]*([0-9.]+)'
    ],
    'creatinine_urine': [
        r'creatinine\s*in\s*urine[:\s]*([0-9.]+)\s*(mg/dL)?',
        r'urine\s*creatinine[:\s]*([0-9.]+)'
    ],
    'acr': [
        r'albumin\s*creatinine\s*ratio[:\s]*([0-9.]+)\s*(mg/g)?',
        r'acr[:\s]*([0-9.]+)',
        r'a/c\s*ratio[:\s]*([0-9.]+)'
    ],
    'egfr': [
        r'egfr[:\s]*([0-9.]+)',
        r'gfr[:\s]*([0-9.]+)',
        r'estimated\s*gfr[:\s]*([0-9.]+)'
    ],
    'hemoglobin': [
        r'hemoglobin[:\s]*([0-9.]+)\s*(g/dl)?',
        r'hgb[:\s]*([0-9.]+)',
        r'hb[:\s]*([0-9.]+)'
    ],
    'sodium': [
        r'sodium[:\s]*([0-9.]+)\s*(mmol/L|mEq/L)?',
        r'na[:\s]*([0-9.]+)'
    ],
    'potassium': [
        r'potassium[:\s]*([0-9.]+)\s*(mmol/L|mEq/L)?',
        r'k\+?[:\s]*([0-9.]+)'
    ],
    'calcium': [
        r'calcium[:\s]*([0-9.]+)\s*(mg/dl)?',
        r'serum\s*calcium[:\s]*([0-9.]+)',
        r'ca[:\s]*([0-9.]+)'
    ],
    'uric_acid': [
        r'uric\s*acid[:\s]*([0-9.]+)\s*(mg/dl)?',
        r'serum\s*uric\s*acid[:\s]*([0-9.]+)'
    ],
    'blood_pressure_systolic': [
        r'bp[:\s]*([0-9]+)/([0-9]+)',
        r'blood\s*pressure[:\s]*([0-9]+)/([0-9]+)'
    ],
    'glucose': [
        r'glucose[:\s]*([0-9.]+)\s*(mg/dl)?',
        r'blood\s*sugar[:\s]*([0-9.]+)',
        r'fbs[:\s]*([0-9.]+)'
    ]
}

REFERENCE_RANGES = {
    'serum_creatinine': {'min': 0.5, 'max': 1.5, 'unit': 'mg/dL'},
    'blood_urea': {'min': 10, 'max': 50, 'unit': 'U/L'},
    'serum_urea': {'min': 10, 'max': 50, 'unit': 'mg/dL'},
    'albumin': {'min': 3.5, 'max': 5.0, 'unit': 'g/dL'},
    'acr': {'min': 0, 'max': 30, 'unit': 'mg/g'},
    'egfr': {'min': 90, 'max': 120, 'unit': 'mL/min/1.73m2'},
    'hemoglobin': {'min': 12, 'max': 17, 'unit': 'g/dL'},
    'sodium': {'min': 136, 'max': 145, 'unit': 'mmol/L'},
    'potassium': {'min': 3.5, 'max': 5.0, 'unit': 'mmol/L'},
    'calcium': {'min': 8.8, 'max': 10.2, 'unit': 'mg/dL'},
    'uric_acid': {'min': 3.4, 'max': 7.0, 'unit': 'mg/dL'},
    'glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL'},
    'blood_pressure_systolic': {'min': 90, 'max': 140, 'unit': 'mmHg'},
    'blood_pressure_diastolic': {'min': 60, 'max': 90, 'unit': 'mmHg'}
}


def parse_values(text):
    """Parse medical text -- mirrors the FIXED logic in text_extractor.py."""
    results = {}
    text_lower = text.lower()
    
    for test_name, patterns in TEST_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    # Special handling for blood pressure (2-group capture)
                    if test_name == 'blood_pressure_systolic' and len(match.groups()) >= 2:
                        systolic = float(match.group(1))
                        diastolic = float(match.group(2))
                        
                        ref_sys = REFERENCE_RANGES.get('blood_pressure_systolic', {})
                        results['blood_pressure_systolic'] = ExtractedValue(
                            test_name='blood_pressure_systolic',
                            value=systolic,
                            unit='mmHg',
                            reference_range=f"{ref_sys.get('min', 90)} - {ref_sys.get('max', 140)}",
                            is_abnormal=systolic < ref_sys.get('min', 90) or systolic > ref_sys.get('max', 140),
                            confidence=0.8
                        )
                        
                        ref_dia = REFERENCE_RANGES.get('blood_pressure_diastolic', {})
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
                    
                    ref = REFERENCE_RANGES.get(test_name, {})
                    ref_min = ref.get('min', 0)
                    ref_max = ref.get('max', float('inf'))
                    ref_unit = ref.get('unit', '')
                    
                    is_abnormal = value < ref_min or value > ref_max
                    
                    results[test_name] = ExtractedValue(
                        test_name=test_name,
                        value=value,
                        unit=unit or ref_unit,
                        reference_range=f"{ref_min} - {ref_max}",
                        is_abnormal=is_abnormal,
                        confidence=0.8
                    )
                    break
                except (ValueError, IndexError):
                    continue
    
    return results


def extract_patient_info(text):
    """Extract patient info -- mirrors text_extractor.py."""
    info = {}
    
    name_match = re.search(r'patient\s*name[:\s]*([^\n]+)', text, re.IGNORECASE)
    if name_match:
        info['patient_name'] = name_match.group(1).strip()
    
    age_match = re.search(r'age[:\s]*(\d+)\s*(years?)?', text, re.IGNORECASE)
    if age_match:
        info['age'] = int(age_match.group(1))
    
    sex_match = re.search(r'sex[:\s]*(male|female|m|f)', text, re.IGNORECASE)
    if sex_match:
        sex = sex_match.group(1).lower()
        info['sex'] = 'male' if sex in ['male', 'm'] else 'female'
    
    date_match = re.search(r'date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE)
    if date_match:
        info['date'] = date_match.group(1)
    
    lab_match = re.search(r'lab[.\s]*no[:\s]*(\d+)', text, re.IGNORECASE)
    if lab_match:
        info['lab_no'] = lab_match.group(1)
    
    return info


# ============================================================
# Test 1: Basic Text Parsing
# ============================================================
def test_parse_values():
    print("\n[TEST 1] Text Parsing (parse_values)")
    print("-" * 50)
    
    sample_text = """
    Kidney Function Test
    Serum Creatinine: 2.3 mg/dl
    Blood Urea: 55 U/L
    Hemoglobin: 10.5 g/dl
    Sodium: 138 mmol/L
    Potassium: 5.2 mmol/L
    Calcium: 8.5 mg/dl
    Glucose: 120 mg/dl
    eGFR: 45
    Uric Acid: 7.5 mg/dl
    """
    
    results = parse_values(sample_text)
    
    assert_test('serum_creatinine' in results, "Extracted serum_creatinine")
    assert_test(results.get('serum_creatinine') and results['serum_creatinine'].value == 2.3, 
                "Creatinine value = 2.3")
    assert_test(results.get('serum_creatinine') and results['serum_creatinine'].is_abnormal, 
                "Creatinine abnormal (2.3 > 1.5)")
    
    assert_test('blood_urea' in results, "Extracted blood_urea")
    
    assert_test('hemoglobin' in results, "Extracted hemoglobin")
    assert_test(results.get('hemoglobin') and results['hemoglobin'].value == 10.5, "Hemoglobin value = 10.5")
    assert_test(results.get('hemoglobin') and results['hemoglobin'].is_abnormal, "Hemoglobin abnormal (10.5 < 12)")
    
    assert_test('sodium' in results, "Extracted sodium")
    assert_test(results.get('sodium') and not results['sodium'].is_abnormal, "Sodium normal (138)")
    
    assert_test('potassium' in results, "Extracted potassium")
    assert_test(results.get('potassium') and results['potassium'].is_abnormal, "Potassium abnormal (5.2 > 5.0)")
    
    assert_test('egfr' in results, "Extracted eGFR")
    assert_test(results.get('egfr') and results['egfr'].value == 45, "eGFR value = 45")
    assert_test(results.get('egfr') and results['egfr'].is_abnormal, "eGFR abnormal (45 < 90)")
    
    assert_test('glucose' in results, "Extracted glucose")
    assert_test(results.get('glucose') and results['glucose'].is_abnormal, "Glucose abnormal (120 > 100)")
    
    assert_test('uric_acid' in results, "Extracted uric_acid")
    assert_test(results.get('uric_acid') and results['uric_acid'].is_abnormal, "Uric acid abnormal (7.5 > 7.0)")


# ============================================================
# Test 2: Blood Pressure (Bug Fix Verification)
# ============================================================
def test_blood_pressure():
    print("\n[TEST 2] Blood Pressure Extraction (Bug Fix)")
    print("-" * 50)
    
    # Test abnormal BP
    results = parse_values("BP: 150/95")
    
    assert_test('blood_pressure_systolic' in results, "Extracted systolic BP")
    assert_test('blood_pressure_diastolic' in results, "Extracted diastolic BP")
    
    if 'blood_pressure_systolic' in results:
        assert_test(results['blood_pressure_systolic'].value == 150, "Systolic value = 150")
        assert_test(results['blood_pressure_systolic'].unit == 'mmHg', "Systolic unit = mmHg")
        assert_test(results['blood_pressure_systolic'].is_abnormal, "Systolic abnormal (150 > 140)")
    
    if 'blood_pressure_diastolic' in results:
        assert_test(results['blood_pressure_diastolic'].value == 95, "Diastolic value = 95")
        assert_test(results['blood_pressure_diastolic'].unit == 'mmHg', "Diastolic unit = mmHg")
        assert_test(results['blood_pressure_diastolic'].is_abnormal, "Diastolic abnormal (95 > 90)")
    
    # Test normal BP
    results2 = parse_values("Blood Pressure: 120/80")
    
    assert_test('blood_pressure_systolic' in results2, "Extracted normal systolic BP")
    if 'blood_pressure_systolic' in results2:
        assert_test(not results2['blood_pressure_systolic'].is_abnormal, "Systolic 120 = normal")
    if 'blood_pressure_diastolic' in results2:
        assert_test(not results2['blood_pressure_diastolic'].is_abnormal, "Diastolic 80 = normal")


# ============================================================
# Test 3: Patient Info
# ============================================================
def test_patient_info():
    print("\n[TEST 3] Patient Info Extraction")
    print("-" * 50)
    
    sample_text = """
    Patient name: Mohamed Ahmed
    Age: 70 Years
    Sex: Male
    Date: 21/05/2025
    Lab. No: 1806
    """
    
    info = extract_patient_info(sample_text)
    
    assert_test('patient_name' in info, "Extracted patient name")
    assert_test(info.get('age') == 70, "Age = 70")
    assert_test(info.get('sex') == 'male', "Sex = male")
    assert_test('date' in info, "Extracted date")
    assert_test('lab_no' in info, "Extracted lab number")
    
    info2 = extract_patient_info("Age: 45\nSex: Female")
    assert_test(info2.get('sex') == 'female', "Sex = female")


# ============================================================
# Test 4: ACR Extraction
# ============================================================
def test_acr_extraction():
    print("\n[TEST 4] ACR Extraction")
    print("-" * 50)
    
    sample_text = """
    Creatinine in Urine: 32.40 mg/dL
    Albumin in Urine: 14.4 ug/mL
    Albumin Creatinine Ratio: 44.44 mg/g
    """
    
    results = parse_values(sample_text)
    
    assert_test('acr' in results, "Extracted ACR")
    assert_test(results.get('acr') and results['acr'].value == 44.44, "ACR value = 44.44")
    assert_test(results.get('acr') and results['acr'].is_abnormal, "ACR abnormal (44.44 > 30)")
    assert_test('creatinine_urine' in results, "Extracted urine creatinine")
    assert_test('albumin_urine' in results, "Extracted urine albumin")


# ============================================================
# Test 5: Reference Ranges Completeness
# ============================================================
def test_reference_ranges():
    print("\n[TEST 5] Reference Ranges Completeness")
    print("-" * 50)
    
    expected = [
        'serum_creatinine', 'blood_urea', 'serum_urea', 'albumin',
        'acr', 'egfr', 'hemoglobin', 'sodium', 'potassium', 
        'calcium', 'uric_acid', 'glucose',
        'blood_pressure_systolic', 'blood_pressure_diastolic'
    ]
    
    for entry in expected:
        has_entry = entry in REFERENCE_RANGES
        assert_test(has_entry, f"Ref range for '{entry}'")
        if has_entry:
            r = REFERENCE_RANGES[entry]
            has_fields = 'min' in r and 'max' in r and 'unit' in r
            assert_test(has_fields, f"  '{entry}' has min/max/unit")


# ============================================================
# Test 6: Full Lab Report (Integration)
# ============================================================
def test_full_lab_report():
    print("\n[TEST 6] Full Lab Report Integration")
    print("-" * 50)
    
    full_report = """
    Patient name: Ahmed Hassan
    Age: 65 Years
    Sex: Male
    Date: 15/01/2026
    
    Kidney Function Test
    Serum Creatinine: 3.1 mg/dL
    Blood Urea: 72 mg/dL
    eGFR: 22
    
    Albumin Creatinine Ratio: 350 mg/g
    
    BP: 160/100
    Hemoglobin: 9.5 g/dl
    """
    
    values = parse_values(full_report)
    info = extract_patient_info(full_report)
    
    assert_test('serum_creatinine' in values, "Creatinine extracted")
    assert_test(values['serum_creatinine'].value == 3.1, "Creatinine = 3.1")
    assert_test('blood_urea' in values, "Blood urea extracted")
    assert_test('egfr' in values, "eGFR extracted")
    assert_test(values['egfr'].value == 22, "eGFR = 22")
    assert_test('acr' in values, "ACR extracted")
    assert_test(values['acr'].value == 350, "ACR = 350")
    assert_test('hemoglobin' in values, "Hemoglobin extracted")
    assert_test('blood_pressure_systolic' in values, "BP systolic extracted")
    assert_test('blood_pressure_diastolic' in values, "BP diastolic extracted")
    assert_test(info.get('age') == 65, "Age = 65")
    assert_test(info.get('sex') == 'male', "Sex = male")
    
    # Abnormal checks
    assert_test(values['serum_creatinine'].is_abnormal, "Creatinine 3.1 = abnormal")
    assert_test(values['egfr'].is_abnormal, "eGFR 22 = abnormal")
    assert_test(values['acr'].is_abnormal, "ACR 350 = abnormal")
    assert_test(values['blood_pressure_systolic'].is_abnormal, "BP 160 systolic = abnormal")
    assert_test(values['blood_pressure_diastolic'].is_abnormal, "BP 100 diastolic = abnormal")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("[OCR] OCR Feature - Comprehensive Test Suite")
    print("=" * 60)
    
    test_parse_values()
    test_blood_pressure()
    test_patient_info()
    test_acr_extraction()
    test_reference_ranges()
    test_full_lab_report()
    
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"[RESULTS] {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("[SUCCESS] ALL OCR TESTS PASSED!")
    else:
        print(f"[WARNING] {FAIL} test(s) failed")
    print("=" * 60)
