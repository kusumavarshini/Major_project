"""
Database module for Fingerprint Recognition and Rectification system.
Handles MySQL connection, schema creation, migrations, and transactional queries.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector
from mysql.connector import errorcode
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv(Path(__file__).resolve().parent / ".env")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "fingerprint_db")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "45.0"))
if MATCH_THRESHOLD <= 1.0:
    # Upgrade legacy fractional 0.0-1.0 threshold to SourceAFIS raw score scale
    MATCH_THRESHOLD = 45.0


def get_connection_config(include_db: bool = True) -> Dict[str, Any]:
    """Get connection dictionary for MySQL connector."""
    cfg = {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
        "charset": "utf8mb4",
        "use_pure": True,
    }
    if include_db:
        cfg["database"] = MYSQL_DATABASE
    return cfg


def test_mysql_connection() -> Tuple[bool, str]:
    """
    Test whether the development environment can connect to MySQL Server.
    Returns (success: bool, message: str).
    """
    try:
        conn = mysql.connector.connect(**get_connection_config(include_db=False))
        server_info = conn.get_server_info()
        conn.close()
        return True, f"Connected successfully to MySQL Server {server_info} at {MYSQL_HOST}:{MYSQL_PORT} as {MYSQL_USER}"
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            return False, f"Access denied for user '{MYSQL_USER}'@'{MYSQL_HOST}'. Please verify MYSQL_PASSWORD in .env."
        elif err.errno == errorcode.CR_CONN_HOST_ERROR or err.errno == 2003:
            return False, f"Cannot connect to MySQL server at {MYSQL_HOST}:{MYSQL_PORT}. Verify that the MySQL service is running."
        else:
            return False, f"MySQL connection error [{err.errno}]: {err.msg}"
    except Exception as e:
        return False, f"Unexpected error connecting to MySQL: {str(e)}"


def get_db_connection():
    """Get active connection to the application database."""
    return mysql.connector.connect(**get_connection_config(include_db=True))


def init_db(confirm_existing: bool = True) -> Dict[str, Any]:
    """
    Initializes the MySQL database and creates tables if they do not exist.
    Does NOT drop existing databases or tables.
    """
    success, msg = test_mysql_connection()
    if not success:
        raise ConnectionError(f"MySQL unavailable: {msg}")

    # 1. Connect without database and create database if not exists
    server_conn = mysql.connector.connect(**get_connection_config(include_db=False))
    cursor = server_conn.cursor()
    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    cursor.close()
    server_conn.close()

    # 2. Connect to the database and create tables
    db_conn = get_db_connection()
    cursor = db_conn.cursor()

    # Table 1: fingerprints
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fingerprints (
            fingerprint_id INT AUTO_INCREMENT PRIMARY KEY,
            subject_id VARCHAR(100) NOT NULL,
            family_id VARCHAR(50),
            member_type VARCHAR(50),
            impression_id INT,
            filename VARCHAR(255) NOT NULL,
            image_path VARCHAR(500) NOT NULL,
            distortion_type VARCHAR(50) DEFAULT 'clean',
            dataset_split VARCHAR(50) DEFAULT 'gallery',
            image_width INT,
            image_height INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uk_file_distortion (filename, distortion_type),
            INDEX idx_subject_id (subject_id),
            INDEX idx_family_id (family_id),
            INDEX idx_distortion_type (distortion_type),
            INDEX idx_dataset_split (dataset_split)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
    )

    # Table 2: fingerprint_templates
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fingerprint_templates (
            template_id INT AUTO_INCREMENT PRIMARY KEY,
            fingerprint_id INT NOT NULL,
            template_data LONGBLOB NOT NULL,
            minutiae_count INT DEFAULT 0,
            template_version VARCHAR(50) DEFAULT 'sourceafis_v1',
            generation_status VARCHAR(50) DEFAULT 'generated',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (fingerprint_id) REFERENCES fingerprints(fingerprint_id) ON DELETE CASCADE,
            INDEX idx_fingerprint_id (fingerprint_id),
            INDEX idx_generation_status (generation_status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
    )

    # Table 3: evaluation_results
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_results (
            evaluation_id INT AUTO_INCREMENT PRIMARY KEY,
            query_fingerprint_id INT NULL,
            reference_fingerprint_id INT NULL,
            original_score FLOAT NOT NULL,
            rectified_score FLOAT NOT NULL,
            original_match BOOLEAN NOT NULL,
            rectified_match BOOLEAN NOT NULL,
            score_diff FLOAT NOT NULL,
            outcome VARCHAR(20) NOT NULL,
            processing_time_ms FLOAT NOT NULL,
            label INT NOT NULL,
            distortion_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_label (label),
            INDEX idx_outcome (outcome),
            INDEX idx_distortion_type (distortion_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
    )

    db_conn.commit()

    # 3. Verify tables exist
    cursor.execute("SHOW TABLES;")
    tables = [row[0] for row in cursor.fetchall()]

    cursor.close()
    db_conn.close()

    return {
        "status": "success",
        "database": MYSQL_DATABASE,
        "tables": tables,
        "message": f"Database '{MYSQL_DATABASE}' successfully initialized with tables: {', '.join(tables)}"
    }


def get_dataset_statistics() -> Dict[str, Any]:
    """
    Retrieve actual statistics from MySQL for the dataset and templates.
    
    NOTE: This queries the application database schema (fingerprint_db).
    For the actual FVC gallery statistics, use get_gallery_statistics().
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT COUNT(*) AS total_fingerprints FROM fingerprints;")
        row_fp = cursor.fetchone() or {"total_fingerprints": 0}

        cursor.execute("SELECT COUNT(DISTINCT subject_id) AS total_subjects FROM fingerprints;")
        row_sub = cursor.fetchone() or {"total_subjects": 0}

        cursor.execute("SELECT COUNT(DISTINCT family_id) AS total_families FROM fingerprints;")
        row_fam = cursor.fetchone() or {"total_families": 0}

        cursor.execute(
            """
            SELECT 
                COUNT(*) AS total_templates,
                SUM(CASE WHEN generation_status = 'generated' THEN 1 ELSE 0 END) AS generated_templates,
                SUM(CASE WHEN generation_status = 'failed' THEN 1 ELSE 0 END) AS failed_templates,
                SUM(CASE WHEN generation_status = 'pending' THEN 1 ELSE 0 END) AS pending_templates
            FROM fingerprint_templates;
            """
        )
        row_tpl = cursor.fetchone() or {}

        cursor.execute(
            """
            SELECT distortion_type, COUNT(*) AS count
            FROM fingerprints
            GROUP BY distortion_type;
            """
        )
        dist_rows = cursor.fetchall()
        distortion_breakdown = {r["distortion_type"]: r["count"] for r in dist_rows}

        cursor.close()
        conn.close()

        return {
            "total_fingerprints": row_fp["total_fingerprints"],
            "total_subjects": row_sub["total_subjects"],
            "total_families": row_fam["total_families"],
            "total_templates": row_tpl.get("total_templates") or 0,
            "generated_templates": int(row_tpl.get("generated_templates") or 0),
            "failed_templates": int(row_tpl.get("failed_templates") or 0),
            "pending_templates": int(row_tpl.get("pending_templates") or 0),
            "distortion_breakdown": distortion_breakdown,
        }
    except Exception as e:
        return {
            "total_fingerprints": 0,
            "total_subjects": 0,
            "total_families": 0,
            "total_templates": 0,
            "generated_templates": 0,
            "failed_templates": 0,
            "pending_templates": 0,
            "distortion_breakdown": {},
            "error": str(e),
        }


def get_gallery_statistics() -> Dict[str, Any]:
    """
    Retrieve statistics from the actual operational FVC gallery (newdb.external_fvc_clean_gallery).
    This is the gallery used by the live identification pipeline.
    """
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database="newdb",
            charset="utf8mb4",
            use_pure=True,
        )
        cursor = conn.cursor(dictionary=True)

        # Gallery templates
        cursor.execute("""
            SELECT COUNT(*) AS total_templates
            FROM external_fvc_clean_gallery
            WHERE generation_status = 'generated' AND template_data IS NOT NULL
        """)
        row_gallery = cursor.fetchone() or {"total_templates": 0}

        cursor.execute("""
            SELECT MIN(fingerprint_id) AS min_id, MAX(fingerprint_id) AS max_id
            FROM external_fvc_clean_gallery
            WHERE generation_status = 'generated' AND template_data IS NOT NULL
        """)
        row_range = cursor.fetchone() or {"min_id": None, "max_id": None}

        # Enrolled fingerprints
        cursor.execute("SELECT COUNT(*) AS enrolled_count FROM enrolled_fingerprints")
        row_enrolled = cursor.fetchone() or {"enrolled_count": 0}

        cursor.execute("SELECT COUNT(DISTINCT subject_id) AS enrolled_subjects FROM enrolled_fingerprints")
        row_subj = cursor.fetchone() or {"enrolled_subjects": 0}

        cursor.close()
        conn.close()

        return {
            "database": "newdb",
            "gallery_table": "external_fvc_clean_gallery",
            "gallery_templates": int(row_gallery["total_templates"]),
            "gallery_id_range": f"{row_range['min_id']} to {row_range['max_id']}" if row_range['min_id'] else "N/A",
            "enrolled_fingerprints": int(row_enrolled["enrolled_count"]),
            "enrolled_subjects": int(row_subj["enrolled_subjects"]),
            "note": "This is the operational gallery used by the live identification pipeline.",
        }
    except Exception as e:
        return {
            "database": "newdb",
            "gallery_templates": 0,
            "enrolled_fingerprints": 0,
            "enrolled_subjects": 0,
            "error": str(e),
        }


if __name__ == "__main__":
    print("Testing MySQL connection...")
    ok, msg = test_mysql_connection()
    print("Result:", ok)
    print("Message:", msg)
    if ok:
        print("\nInitializing database schema...")
        res = init_db()
        print("Initialization:", res)
        print("\nCurrent dataset statistics:")
        stats = get_dataset_statistics()
        print(stats)
