from pathlib import Path
import sys
import json
import time
import pickle
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ============================================================
# PROJECT IMPORTS
# ============================================================

import authentication as auth
from matcher import SourceAFISMatcher
import mysql.connector


# ============================================================
# PATHS
# ============================================================

LATENT_DIR = ROOT / "data" / "Latent_fvc"
CLEAN_DIR = ROOT / "data" / "clean_fvc_new"

CLASSIFIER_CHECKPOINT = (
    ROOT / "experiments" / "classifier_fvc_adapted" / "best_model.pth"
)

CLASSIFIER_THRESHOLD_FILE = (
    ROOT / "experiments" / "classifier_fvc_adapted" / "threshold.json"
)

DDRN_CHECKPOINT = Path(
    r"C:\Users\kusum\OneDrive\Desktop\fingerprint"
    r"\experiments\ddrnet_baseline\best_model.pth"
)

OUTPUT_CSV = ROOT / "latent_genuine_pair_evaluation.csv"


# ============================================================
# MYSQL CONFIGURATION
# SAME CONFIGURATION AS WORKING 317 1:N EVALUATOR
# ============================================================

DB_NAME = "newDB"
DB_TABLE = "external_fvc_clean_gallery"

MYSQL_HOST = os.getenv(
    "MYSQL_HOST",
    "localhost"
)

MYSQL_PORT = int(
    os.getenv(
        "MYSQL_PORT",
        "3306"
    )
)

MYSQL_USER = os.getenv(
    "MYSQL_USER",
    "root"
)

MYSQL_PASSWORD = os.getenv(
    "MYSQL_PASSWORD",
    "1234"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# STARTUP INFORMATION
# ============================================================
