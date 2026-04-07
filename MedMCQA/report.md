# MedMCQA Dataset - Subject Split Report

## Overview

- **Total questions in dataset**: 193,155
- **Valid questions (in known subjects)**: 189,426
- **Filtered out questions**: 3,729
- **Number of subjects**: 20

### Questions per Source Split

| Split | Total | Valid | Filtered |
|-------|------:|------:|---------:|
| dev | 4,183 | 4,181 | 2 |
| test | 6,150 | 5,468 | 682 |
| train | 182,822 | 179,777 | 3,045 |

### Filtered Out Questions Detail

| Subject Name | Count |
|-------------|------:|
| Unknown | 3,729 |

## Per-Subject Breakdown

| # | Subject (Dataset) | Display Name | Dev | Test | Train | **Total** | Single | Multi | With Exp | Without Exp | File |
|--:|-------------------|-------------|----:|-----:|------:|----------:|-------:|------:|---------:|------------:|------|
| 1 | Anaesthesia | Anesthesia | 34 | 59 | 3,172 | **3,265** | 2,125 | 1,140 | 3,106 | 159 | `Anaesthesia.json` |
| 2 | Anatomy | Anatomy | 234 | 259 | 14,560 | **15,053** | 9,561 | 5,492 | 14,037 | 1,016 | `Anatomy.json` |
| 3 | Biochemistry | Biochemistry | 171 | 352 | 8,282 | **8,805** | 6,340 | 2,465 | 8,010 | 795 | `Biochemistry.json` |
| 4 | Dental | Dental | 1,318 | 1,203 | 8,938 | **11,459** | 7,240 | 4,219 | 4,189 | 7,270 | `Dental.json` |
| 5 | ENT | ENT | 53 | 86 | 4,919 | **5,058** | 3,305 | 1,753 | 4,441 | 617 | `ENT.json` |
| 6 | Forensic Medicine | Forensic Medicine (FM) | 67 | 132 | 5,900 | **6,099** | 4,401 | 1,698 | 5,818 | 281 | `Forensic_Medicine.json` |
| 7 | Gynaecology & Obstetrics | Obstetrics and Gynecology (O&G) | 224 | 532 | 10,013 | **10,769** | 6,915 | 3,854 | 9,874 | 895 | `Gynaecology_Obstetrics.json` |
| 8 | Medicine | Medicine | 295 | 372 | 17,887 | **18,554** | 11,345 | 7,209 | 14,751 | 3,803 | `Medicine.json` |
| 9 | Microbiology | Microbiology | 122 | 167 | 11,314 | **11,603** | 7,808 | 3,795 | 10,142 | 1,461 | `Microbiology.json` |
| 10 | Ophthalmology | Ophthalmology | 58 | 177 | 6,932 | **7,167** | 4,783 | 2,384 | 6,897 | 270 | `Ophthalmology.json` |
| 11 | Orthopaedics | Orthopedics | 20 | 0 | 2,999 | **3,019** | 2,090 | 929 | 2,770 | 249 | `Orthopaedics.json` |
| 12 | Pathology | Pathology | 337 | 305 | 14,884 | **15,526** | 10,421 | 5,105 | 13,095 | 2,431 | `Pathology.json` |
| 13 | Pediatrics | Pediatrics | 234 | 190 | 8,037 | **8,461** | 5,627 | 2,834 | 7,854 | 607 | `Pediatrics.json` |
| 14 | Pharmacology | Pharmacology | 243 | 317 | 13,758 | **14,318** | 9,840 | 4,478 | 11,422 | 2,896 | `Pharmacology.json` |
| 15 | Physiology | Physiology | 171 | 388 | 8,830 | **9,389** | 6,639 | 2,750 | 8,438 | 951 | `Physiology.json` |
| 16 | Psychiatry | Psychiatry | 16 | 6 | 4,442 | **4,464** | 2,767 | 1,697 | 4,323 | 141 | `Psychiatry.json` |
| 17 | Radiology | Radiology | 69 | 119 | 4,395 | **4,583** | 3,283 | 1,300 | 4,085 | 498 | `Radiology.json` |
| 18 | Skin | Skin | 17 | 60 | 1,771 | **1,848** | 1,330 | 518 | 1,788 | 60 | `Skin.json` |
| 19 | Social & Preventive Medicine | Preventive & Social Medicine (PSM) | 129 | 243 | 11,882 | **12,254** | 8,317 | 3,937 | 10,476 | 1,778 | `Social_Preventive_Medicine.json` |
| 20 | Surgery | Surgery | 369 | 501 | 16,862 | **17,732** | 11,125 | 6,607 | 14,581 | 3,151 | `Surgery.json` |
| | **TOTAL** | | **4,181** | **5,468** | **179,777** | **189,426** | **125,262** | **64,164** | **160,097** | **29,329** | |

## Data Fields

Each question in the output JSON files contains the following fields:

| Field | Description |
|-------|-------------|
| `id` | Unique string identifier for the question |
| `question` | Question text |
| `opa` | Option A |
| `opb` | Option B |
| `opc` | Option C |
| `opd` | Option D |
| `cop` | Correct option (1=A, 2=B, 3=C, 4=D) — present in dev and train splits |
| `choice_type` | `single` or `multi` choice question |
| `exp` | Expert's explanation (may be null) |
| `subject_name` | Medical subject name |
| `topic_name` | Medical topic name (may be null) |
| `split` | Original split: `dev`, `test`, or `train` |

## Output Structure

```
MedMCQA/
├── data/
│   ├── dev.json           (original)
│   ├── test.json          (original)
│   ├── train.json         (original)
│   └── by_subject/
│       ├── Anaesthesia.json  (3,265 questions)
│       ├── Anatomy.json  (15,053 questions)
│       ├── Biochemistry.json  (8,805 questions)
│       ├── Dental.json  (11,459 questions)
│       ├── ENT.json  (5,058 questions)
│       ├── Forensic_Medicine.json  (6,099 questions)
│       ├── Gynaecology_Obstetrics.json  (10,769 questions)
│       ├── Medicine.json  (18,554 questions)
│       ├── Microbiology.json  (11,603 questions)
│       ├── Ophthalmology.json  (7,167 questions)
│       ├── Orthopaedics.json  (3,019 questions)
│       ├── Pathology.json  (15,526 questions)
│       ├── Pediatrics.json  (8,461 questions)
│       ├── Pharmacology.json  (14,318 questions)
│       ├── Physiology.json  (9,389 questions)
│       ├── Psychiatry.json  (4,464 questions)
│       ├── Radiology.json  (4,583 questions)
│       ├── Skin.json  (1,848 questions)
│       ├── Social_Preventive_Medicine.json  (12,254 questions)
│       ├── Surgery.json  (17,732 questions)
├── split_by_subject.py
└── report.md
```
