
#!/usr/bin/env python3
"""
Enroll New Fingerprint Using Physical SecuGen Hamster Pro 20 Scanner
Standalone terminal-driven procedure to capture a fresh physical fingerprint
and enroll it safely into the existing MySQL database (newdb.enrolled_fingerprints).
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import base64
import ssl
import json
import urllib.request
import urllib.parse
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcher import MATCHER
from mysql_auth import enroll_fingerprint, get_person_templates, get_all_gallery_templates, clear_gallery_cache
from database import test_mysql_connection
from secugen_scanner import SecuGenNativeScanner

# SecuGen Configuration (WebAPI Fallback)
SECUGEN_PORTS = [8443, 8000]
CAPTURE_ENDPOINT = "/SGIFPCapture"


def check_secugen_service() -> tuple[str, int] | None:
    """Find active SecuGen WebAPI service."""
    import socket
    for port in SECUGEN_PORTS:
        proto = "https" if port == 8443 else "http"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.5)
        res = sock.connect_ex(("localhost", port))
        sock.close()
        if res == 0:
            return f"{proto}://localhost:{port}", port
    return None


def capture_from_scanner(service_url: str, timeout_sec: int = 25) -> dict:
    """Trigger physical capture from SecuGen Hamster Pro 20 via WebAPI."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    license_str = os.getenv("SECUGEN_LIC", "")

    params = {
        "Timeout": str(timeout_sec * 1000),
        "Quality": "50",
        "licstr": license_str,
        "fakeDetection": "0",
        "templateFormat": "0",
        "imageWSQRate": "0.75",
    }

    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(f"{service_url}{CAPTURE_ENDPOINT}", data=data)

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout_sec + 5.0) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content)
    except Exception as e:
        raise ConnectionError(f"Failed to communicate with SecuGen WebAPI at {service_url}: {e}")


def main():
    print("=" * 70)
    print("   SECUGEN HAMSTER PRO 20 - LIVE FINGERPRINT ENROLLMENT")
    print("=" * 70)

    # 1. Check MySQL connection
    print("\n[Step 1/6] Checking MySQL database connectivity...")
    ok, msg = test_mysql_connection()
    if not ok:
        print(f"❌ MySQL Error: {msg}")
        sys.exit(1)
    print("✅ MySQL connected successfully.")

    # 2. Check SecuGen Hardware & Service
    print("\n[Step 2/6] Detecting SecuGen Scanner...")
    native_scanner = None
    service_url = None

    try:
        scanner = SecuGenNativeScanner()
        dev_info = scanner.open()
        serial_str = dev_info.DeviceSN.decode(errors='ignore').strip()
        print(f"✅ SecuGen Native Hardware detected!")
        print(f"   Device Serial: {serial_str}")
        print(f"   Resolution: {dev_info.ImageWidth} x {dev_info.ImageHeight} pixels @ {dev_info.ImageDPI} DPI")
        native_scanner = scanner
    except Exception as e:
        print(f"ℹ️ Native SDK mode not accessible ({e}), falling back to WebAPI service...")
        service_info = check_secugen_service()
        if not service_info:
            print("❌ SecuGen scanner hardware / WebAPI service is not available.")
            print("   Please connect the Hamster Pro 20 scanner USB cable and retry.")
            sys.exit(1)
        service_url, port = service_info
        print(f"✅ SecuGen Bio Server detected at: {service_url}")

    # 3. Prompt for unique Subject ID
    print("\n[Step 3/6] Subject Identification")
    while True:
        try:
            subject_id = input("Enter unique Subject/Member ID to enroll (e.g., SUBJECT-001): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEnrollment cancelled.")
            if native_scanner:
                native_scanner.close()
            sys.exit(0)

        if not subject_id:
            print("⚠️  Subject ID cannot be empty.")
            continue

        # Check if subject is already enrolled
        try:
            existing = get_person_templates(subject_id)
            if existing and existing.get("templates"):
                print(f"⚠️  Subject ID '{subject_id}' is ALREADY enrolled in the database.")
                print("   Please choose a different, unique identifier.")
                continue
        except ValueError:
            # Not found is expected for new enrollment!
            break
        except Exception as e:
            # Other errors
            print(f"Database check warning: {e}")
            break

    print(f"✅ Target Subject ID: '{subject_id}'")

    # 4. Prompt to capture from physical scanner
    print("\n[Step 4/6] Physical Fingerprint Capture")
    
    image = None
    template = None
    img_h, img_w = 0, 0
    timeout_sec = 25

    while True:
        print("-" * 70)
        print("👉 INSTRUCTIONS:")
        print("   1. Ensure the Hamster Pro 20 USB scanner is firmly plugged in.")
        print("   2. Place your finger firmly and flatly covering the center of the glass prism.")
        print("   3. Keep your finger steady until capture finishes.")
        print("-" * 70)
        
        try:
            input(f"Press [ENTER] when ready to capture (waiting up to {timeout_sec}s)... ")
        except (KeyboardInterrupt, EOFError):
            print("\nEnrollment cancelled.")
            if native_scanner:
                native_scanner.close()
            sys.exit(0)

        print(f"\n📡 Sensor active (waiting up to {timeout_sec} seconds for finger contact)...")
        print("   👉 Red optical LED should be ON. Hold finger firmly on the glass prism!")

        if native_scanner:
            try:
                image, quality = native_scanner.capture_image(timeout_sec=timeout_sec, quality_min=30)
                img_h, img_w = image.shape
                print(f"✅ Fingerprint captured successfully via Native Hardware!")
                print(f"   Image Resolution: {img_w} x {img_h} pixels")
                print(f"   Reported Quality: {quality}/100")
            except TimeoutError:
                print("\n❌ Scanner Capture Failed: Capture timeout - sensor did not detect sufficient finger contact.")
                print("\n💡 Troubleshooting Tips:")
                print("   • Place finger FLAT and centered on the glass prism (cover the center).")
                print("   • Apply firm, even pressure — optical sensors require physical ridge contact.")
                print("   • If finger skin is dry, gently rub finger against palm or breathe warm air on it.")
                print("   • Ensure the prism glass is clean and dry.")
                retry = input("\nWould you like to try capturing again? [Y/n]: ").strip().lower()
                if retry in ("", "y", "yes"):
                    continue
                else:
                    print("Enrollment aborted by user.")
                    native_scanner.close()
                    sys.exit(1)
            except Exception as e:
                print(f"\n❌ Capture error: {e}")
                retry = input("\nWould you like to try capturing again? [Y/n]: ").strip().lower()
                if retry in ("", "y", "yes"):
                    continue
                native_scanner.close()
                sys.exit(1)
        else:
            # Fallback to WebAPI
            try:
                resp = capture_from_scanner(service_url, timeout_sec=timeout_sec)
            except Exception as e:
                print(f"❌ Capture communication error: {e}")
                retry = input("\nWould you like to retry capture? [Y/n]: ").strip().lower()
                if retry in ("", "y", "yes"):
                    continue
                sys.exit(1)

            error_code = resp.get("ErrorCode", -1)
            if error_code != 0:
                error_msgs = {
                    1: "Scanner not found or not connected. Check USB cable.",
                    2: "Capture timeout - finger was not detected on the scanner.",
                    3: "Poor image quality - sensor was unable to get a clear print.",
                    4: "License validation failed. Check SECUGEN_LIC in .env.",
                    10004: "Capture timeout - sensor did not detect sufficient finger contact.",
                }
                msg = error_msgs.get(error_code, f"SecuGen Error Code: {error_code}")
                print(f"\n❌ Scanner Capture Failed: {msg}")
                print("\n💡 Troubleshooting Tips:")
                print("   • Place finger FLAT and centered on the glass prism.")
                print("   • Apply firm, even pressure.")
                print("   • If finger skin is dry, breathe warm air on it.")
                
                retry = input("\nWould you like to try capturing again? [Y/n]: ").strip().lower()
                if retry in ("", "y", "yes"):
                    continue
                else:
                    print("Enrollment aborted by user.")
                    sys.exit(1)

            bmp_base64 = resp.get("BMPBase64")
            if not bmp_base64:
                print("❌ No image data returned by scanner.")
                retry = input("\nWould you like to retry capture? [Y/n]: ").strip().lower()
                if retry in ("", "y", "yes"):
                    continue
                sys.exit(1)

            try:
                image_bytes = base64.b64decode(bmp_base64)
                image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None or image.size == 0:
                    raise ValueError("Decoded image is empty.")
            except Exception as e:
                print(f"❌ Failed to decode captured image: {e}")
                retry = input("\nWould you like to retry capture? [Y/n]: ").strip().lower()
                if retry in ("", "y", "yes"):
                    continue
                sys.exit(1)

            img_h, img_w = image.shape
            quality = resp.get("ImageQuality", "N/A")
            print(f"✅ Fingerprint captured successfully!")
            print(f"   Image Resolution: {img_w} x {img_h} pixels")
            print(f"   Reported Quality: {quality}/100")

        # 5. Extract SourceAFIS template and validate minutiae
        print("\n[Step 5/6] Extracting SourceAFIS Biometric Template...")
        template = MATCHER.extract_template(image)
        minutiae_count = template.get("minutiae_count", 0)
        print(f"   Minutiae count detected: {minutiae_count}")

        if minutiae_count < 5:
            print(f"❌ Insufficient minutiae detected ({minutiae_count} < 5).")
            print("   Fingerprint quality is too low for reliable matching. Please recapture.")
            retry = input("\nRecapture fingerprint? [Y/n]: ").strip().lower()
            if retry in ("", "y", "yes"):
                continue
            sys.exit(1)

        print("✅ Biometric template successfully extracted and verified.")
        break

    # Save original capture image to disk
    image_dir = PROJECT_ROOT / "experiments" / "enrollment_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    filename = f"enrolled_{subject_id}_{uuid.uuid4().hex[:8]}.png"
    image_path = image_dir / filename
    cv2.imwrite(str(image_path), image)
    print(f"   Archived image: {image_path.name}")

    # Confirm enrollment
    confirm = input(f"\nProceed with enrolling '{subject_id}' into database? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes"):
        print("Enrollment cancelled by user.")
        image_path.unlink(missing_ok=True)
        sys.exit(0)

    # 6. Database Insertion
    print("\n[Step 6/6] Inserting record into MySQL...")
    try:
        enrollment_id = enroll_fingerprint(
            subject_id=subject_id,
            filename=filename,
            image_path=str(image_path.resolve()),
            image_width=int(img_w),
            image_height=int(img_h),
            template=template,
        )
        print(f"🎉 SUCCESS! Fingerprint enrolled with Enrollment ID: {enrollment_id}")
    except Exception as e:
        image_path.unlink(missing_ok=True)
        print(f"❌ Enrollment failed during database transaction: {e}")
        sys.exit(1)

    # 7. Verification: Confirm searchability immediately
    print("\n[Verification] Validating immediate searchability in gallery...")
    gallery = get_all_gallery_templates(force_refresh=True)
    enrolled_record = next((g for g in gallery if g.get("subject_id") == subject_id), None)

    if enrolled_record:
        print(f"✅ Confirmed: '{subject_id}' is now present in active gallery cache.")
        print(f"   Total searchable templates in gallery: {len(gallery)}")
        
        # Test self-match
        match_test = MATCHER.identify_1toN(template, gallery, top_n=1)
        if match_test.get("match_found") and match_test.get("best_candidate", {}).get("subject_id") == subject_id:
            print(f"✅ Biometric Self-Match Verification PASSED: Score = {match_test['best_candidate']['raw_score']:.1f}")
        else:
            print("⚠️  Warning: Record inserted but self-match score fell below threshold.")
    else:
        print("⚠️  Warning: Record committed to DB but could not be found in refreshed gallery cache.")

    if native_scanner:
        native_scanner.close()

    print("\n" + "=" * 70)
    print("ENROLLMENT COMPLETE")
    print(f"Subject: {subject_id}")
    print(f"Database: newdb.enrolled_fingerprints (ID: {enrollment_id})")
    print("The new fingerprint is immediately available for 1:N identification.")
    print("=" * 70)


if __name__ == "__main__":
    main()

