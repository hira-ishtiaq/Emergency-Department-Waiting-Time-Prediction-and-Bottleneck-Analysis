"""
nhamcs.py - loader and cleaner for the REAL dataset used in this project:

    National Hospital Ambulatory Medical Care Survey (NHAMCS), 2019 Emergency Department public-use file
    National Center for Health Statistics (NCHS), U.S. Centers for Disease Control and Prevention (CDC).
    Documentation: "2019 NHAMCS Micro-Data File Documentation" (doc19-ed-508.pdf)
    Download page: NCHS NHAMCS datasets page (https://www.cdc.gov/nchs/ahcd/index.htm) - check the exact link you used.

The raw file is fixed-width text (19,481 visits x 2,380 characters). Only the columns listed in FIELDS are read;
their (start, end) character positions (1-based, inclusive) come from the codebook in the documentation PDF.

Special codes in the file: -9 = blank, -8 = unknown, -7 = not applicable. They are converted to missing values.
"""
import os

import numpy as np
import pandas as pd

FIELDS = {
    "VMONTH": (1, 2),
    "VDAYR": (3, 3),
    "ARRTIME": (4, 7),
    "WAITTIME": (8, 11),
    "LOV": (12, 15),
    "AGE": (16, 18),
    "SEX": (25, 25),
    "RESIDNCE": (23, 24),
    "ARREMS": (33, 34),
    "PAYTYPER": (46, 47),
    "TEMPF": (48, 51),
    "PULSE": (52, 54),
    "RESPR": (55, 57),
    "BPSYS": (58, 60),
    "BPDIAS": (61, 63),
    "POPCT": (64, 66),
    "IMMEDR": (67, 68),
    "PAINSCALE": (69, 70),
    "RFV1": (73, 77),
    "INJURY": (100, 101),
    "ABG": (178, 178),
    "BUNCREAT": (182, 182),
    "CARDENZ": (183, 183),
    "CBC": (184, 184),
    "CMP": (185, 185),
    "DDIMER": (191, 191),
    "ELECTROL": (192, 192),
    "GLUCOSE": (193, 193),
    "OTHERBLD": (197, 197),
    "URINE": (204, 204),
    "ANYIMAGE": (206, 206),
    "ADMIT": (501, 502),
    "REGION": (626, 626),
    "MSA": (627, 627),
    "PATWT": (2357, 2367),
}

LAB_COLS = ["ABG", "BUNCREAT", "CARDENZ", "CBC", "CMP", "DDIMER", "ELECTROL", "GLUCOSE", "OTHERBLD", "URINE"]
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "data", "ed2019")
CLEAN_PATH = os.path.join(HERE, "data", "nhamcs2019_clean.csv")


def load_raw(path=RAW_PATH):
    colspecs = [(a - 1, b) for a, b in FIELDS.values()]
    df = pd.read_fwf(path, colspecs=colspecs, names=list(FIELDS), header=None, dtype=str)
    return df.apply(pd.to_numeric, errors="coerce")


def _valid(s, lo, hi):
    return s.where(s.between(lo, hi))


def clean(raw):
    """Turn raw codes into analysis-ready columns. Rows without a recorded waiting time are dropped
    (WAITTIME = -9 blank, or -7 = patient was not seen by a physician/APRN/PA)."""
    d = pd.DataFrame(index=raw.index)
    d["wait_min"] = raw["WAITTIME"].where(raw["WAITTIME"] >= 0)
    d["visit_length_min"] = raw["LOV"].where(raw["LOV"] >= 0)
    d["triage_level"] = _valid(raw["IMMEDR"], 1, 5)            # 1 = immediate ... 5 = non-urgent (NaN = not recorded)
    d["triage_recorded"] = d["triage_level"].notna().astype(int)
    d["age"] = _valid(raw["AGE"], 0, 95)
    d["sex"] = raw["SEX"].map({1: "female", 2: "male"})
    arr = raw["ARRTIME"].where(raw["ARRTIME"].between(0, 2359))
    d["arrival_hour"] = (arr // 100).where(arr % 100 < 60)
    d["day_of_week"] = _valid(raw["VDAYR"], 1, 7)               # 1 = Sunday
    d["is_weekend"] = d["day_of_week"].isin([1, 7]).astype(int)
    d["month"] = _valid(raw["VMONTH"], 1, 12)
    d["ambulance"] = (raw["ARREMS"] == 1).astype(int)
    d["temp_f"] = (raw["TEMPF"] / 10.0).where(raw["TEMPF"].between(870, 1100))
    d["pulse"] = _valid(raw["PULSE"], 20, 250)
    d["resp_rate"] = _valid(raw["RESPR"], 4, 80)
    d["bp_sys"] = _valid(raw["BPSYS"], 50, 280)
    d["bp_dias"] = _valid(raw["BPDIAS"], 20, 180)               # 998 (doppler/palpation) becomes missing
    d["spo2"] = _valid(raw["POPCT"], 50, 100)
    d["pain"] = _valid(raw["PAINSCALE"], 0, 10)
    d["injury"] = (raw["INJURY"] > 0).astype(int)
    d["payer"] = raw["PAYTYPER"].where(raw["PAYTYPER"] > 0)
    d["residence"] = raw["RESIDNCE"].where(raw["RESIDNCE"] > 0)
    d["complaint_group"] = (raw["RFV1"] // 1000).where(raw["RFV1"] > 0)   # first two digits of reason-for-visit code
    d["region"] = _valid(raw["REGION"], 1, 4)
    d["metro"] = raw["MSA"].map({1: 1, 2: 0})                   # 1 = metropolitan area
    d["any_lab"] = (raw[LAB_COLS] == 1).any(axis=1).astype(int)
    d["any_imaging"] = (raw["ANYIMAGE"] == 1).astype(int)
    d["admitted"] = raw["ADMIT"].isin([1, 2, 3, 4, 5, 6]).astype(int)
    d["weight"] = raw["PATWT"]
    return d


def build_clean_csv():
    raw = load_raw()
    d = clean(raw)
    d.to_csv(CLEAN_PATH, index=False)
    return d


def load_clean():
    """Cleaned visits (all 19,481). Uses data/nhamcs2019_clean.csv; rebuilds it from the raw file if needed."""
    if os.path.exists(CLEAN_PATH):
        return pd.read_csv(CLEAN_PATH)
    if os.path.exists(RAW_PATH):
        return build_clean_csv()
    raise SystemExit("Put the NHAMCS file 'ed2019' in the data/ folder (or keep data/nhamcs2019_clean.csv).")


def modelling_table(d=None):
    """Visits that have a valid waiting time: the table used for EDA and machine learning."""
    d = load_clean() if d is None else d
    return d[d["wait_min"].notna()].reset_index(drop=True)


if __name__ == "__main__":
    d = build_clean_csv()
    print(d.shape, "->", CLEAN_PATH)
    print(d.isna().mean().round(3).to_string())
