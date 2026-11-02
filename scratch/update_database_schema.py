"""
Script to safely update database.py with physical dataset collection schema.
"""

from pathlib import Path

ADDITIONAL_CODE = '''

# ==============================================================================
# PHYSICAL DATASET COLLECTION SCHEMA & HELPERS
# ==============================================================================

def init_physical_collection_db():
    """
    Initializes dedicated tables for physical dataset collection:
    - physical_dataset_collection: stores all 10-finger clean and distorted records
    - physical_dataset_failed_captures: logs failed/rejected scan attempts
    Preserves all existing tables (fingerprint_gallery, etc.) non-destructively.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Table 1: physical_dataset_collection
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS physical_dataset_collection (
                id INT AUTO_INCREMENT PRIMARY KEY,
                participant_id VARCHAR(50) NOT NULL,
                hand VARCHAR(10) NOT NULL,
                finger VARCHAR(20) NOT NULL,
                `condition` TINYINT NOT NULL,
                filename VARCHAR(255) NOT NULL,
                image_path VARCHAR(500) NOT NULL,
                capture_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                image_width INT NOT NULL,
                image_height INT NOT NULL,
                capture_status VARCHAR(50) NOT NULL DEFAULT 'accepted',
                scanner_model VARCHAR(100),
                scanner_serial VARCHAR(100),
                image_quality INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_participant_hand_finger_cond (participant_id, hand, finger, `condition`),
                INDEX idx_participant (participant_id),
                INDEX idx_condition (`condition`),
                INDEX idx_hand_finger (hand, finger)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
        )

        # Table 2: physical_dataset_failed_captures
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS physical_dataset_failed_captures (
                id INT AUTO_INCREMENT PRIMARY KEY,
                participant_id VARCHAR(50) NOT NULL,
                hand VARCHAR(10) NOT NULL,
                finger VARCHAR(20) NOT NULL,
                `condition` TINYINT NOT NULL,
                error_message TEXT,
                capture_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                scanner_model VARCHAR(100),
                scanner_serial VARCHAR(100),
                INDEX idx_fail_participant (participant_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
        )

        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success", "message": "Physical collection tables verified/created."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def save_physical_capture_to_db(record_data: dict) -> bool:
    """
    Insert or update a physical fingerprint capture record in MySQL.
    Uses ON DUPLICATE KEY UPDATE to prevent duplicates while allowing updates.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            INSERT INTO physical_dataset_collection (
                participant_id, hand, finger, `condition`, filename,
                image_path, capture_timestamp, image_width, image_height,
                capture_status, scanner_model, scanner_serial, image_quality
            ) VALUES (
                %(participant_id)s, %(hand)s, %(finger)s, %(condition)s, %(filename)s,
                %(image_path)s, %(capture_timestamp)s, %(image_width)s, %(image_height)s,
                %(capture_status)s, %(scanner_model)s, %(scanner_serial)s, %(image_quality)s
            )
            ON DUPLICATE KEY UPDATE
                filename = VALUES(filename),
                image_path = VALUES(image_path),
                capture_timestamp = VALUES(capture_timestamp),
                image_width = VALUES(image_width),
                image_height = VALUES(image_height),
                capture_status = VALUES(capture_status),
                scanner_model = VALUES(scanner_model),
                scanner_serial = VALUES(scanner_serial),
                image_quality = VALUES(image_quality),
                updated_at = CURRENT_TIMESTAMP;
        """
        cursor.execute(query, record_data)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB Warning] Could not save physical capture to MySQL: {e}")
        return False


def record_failed_physical_capture_to_db(fail_data: dict) -> bool:
    """Log a failed or rejected capture attempt in MySQL."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            INSERT INTO physical_dataset_failed_captures (
                participant_id, hand, finger, `condition`, error_message,
                scanner_model, scanner_serial
            ) VALUES (
                %(participant_id)s, %(hand)s, %(finger)s, %(condition)s, %(error_message)s,
                %(scanner_model)s, %(scanner_serial)s
            );
        """
        cursor.execute(query, fail_data)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB Warning] Could not log failed capture to MySQL: {e}")
        return False


def get_physical_collection_stats_from_db() -> dict:
    """Retrieve physical dataset collection statistics from MySQL."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT COUNT(*) AS total_images FROM physical_dataset_collection WHERE capture_status = 'accepted';")
        total_images = (cursor.fetchone() or {}).get("total_images", 0)

        cursor.execute("SELECT COUNT(DISTINCT participant_id) AS total_participants FROM physical_dataset_collection WHERE capture_status = 'accepted';")
        total_participants = (cursor.fetchone() or {}).get("total_participants", 0)

        cursor.execute("SELECT COUNT(*) AS clean_count FROM physical_dataset_collection WHERE `condition` = 0 AND capture_status = 'accepted';")
        clean_count = (cursor.fetchone() or {}).get("clean_count", 0)

        cursor.execute("SELECT COUNT(*) AS dist_count FROM physical_dataset_collection WHERE `condition` = 1 AND capture_status = 'accepted';")
        dist_count = (cursor.fetchone() or {}).get("dist_count", 0)

        cursor.execute("SELECT COUNT(*) AS total_failures FROM physical_dataset_failed_captures;")
        total_failures = (cursor.fetchone() or {}).get("total_failures", 0)

        cursor.close()
        conn.close()

        return {
            "total_images": total_images,
            "total_participants": total_participants,
            "clean_images": clean_count,
            "distorted_images": dist_count,
            "failed_captures": total_failures,
        }
    except Exception as e:
        return {
            "total_images": 0,
            "total_participants": 0,
            "clean_images": 0,
            "distorted_images": 0,
            "failed_captures": 0,
            "error": str(e),
        }
'''

db_path = Path(r"c:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\database.py")
text = db_path.read_text(encoding="utf-8")

if "init_physical_collection_db" not in text:
    pos = text.find("if __name__ == '__main__':")
    if pos == -1:
        pos = text.find('if __name__ == "__main__":')
    if pos != -1:
        new_text = text[:pos] + ADDITIONAL_CODE.strip() + "\n\n\n" + text[pos:]
        db_path.write_text(new_text, encoding="utf-8")
        print("Successfully updated database.py")
    else:
        print("ERROR: Could not find main block in database.py")
else:
    print("Already updated.")

