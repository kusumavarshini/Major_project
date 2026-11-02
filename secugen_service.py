"""
SecuGen Fingerprint Scanner Service
Unified interface for SecuGen Hamster Pro 20:
- Native SDK driver (sgfplib.dll)
- WebAPI Bio Server (https://localhost:8443 / http://localhost:8000)
- Image decoding, validation, and quality assessment
"""

from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from secugen_scanner import SecuGenNativeScanner

SECUGEN_PORTS = [8443, 8000]
CAPTURE_ENDPOINT = "/SGIFPCapture"


@dataclass
class ScannerDeviceStatus:
    available: bool
    mode: str  # "native_hardware", "webapi", "not_connected", "mock"
    device_name: str
    serial_number: str
    resolution: str
    service_url: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class CaptureResult:
    success: bool
    image: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0
    quality: int = 0
    scanner_model: str = ""
    serial_number: str = ""
    capture_method: str = ""
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    is_mock: bool = False


class SecuGenService:
    """Service class for SecuGen scanner detection and capture."""

    def __init__(self):
        self.last_status: Optional[ScannerDeviceStatus] = None

    def check_webapi_port(self) -> Optional[Tuple[str, int]]:
        """Check if SecuGen WebAPI service is responding on port 8443 or 8000."""
        for port in SECUGEN_PORTS:
            proto = "https" if port == 8443 else "http"
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.5)
            res = sock.connect_ex(("localhost", port))
            sock.close()
            if res == 0:
                return f"{proto}://localhost:{port}", port
        return None

    def detect_device(self) -> ScannerDeviceStatus:
        """
        Detect connected SecuGen scanner.
        Checks Native SDK first, then WebAPI service.
        """
        # 1. Check Native Hardware SDK first
        try:
            scanner = SecuGenNativeScanner()
            info = scanner.open()
            serial = info.DeviceSN.decode(errors="ignore").strip()
            w, h, dpi = int(info.ImageWidth), int(info.ImageHeight), int(info.ImageDPI)
            scanner.close()

            status = ScannerDeviceStatus(
                available=True,
                mode="native_hardware",
                device_name="SecuGen Hamster Pro 20",
                serial_number=serial or "Unknown",
                resolution=f"{w}x{h} @ {dpi} DPI",
                service_url=None,
                error_message=None,
            )
            self.last_status = status
            return status
        except Exception as e:
            native_err = str(e)

        # 2. Check WebAPI Bio Server
        webapi_info = self.check_webapi_port()
        if webapi_info:
            url, port = webapi_info
            status = ScannerDeviceStatus(
                available=True,
                mode="webapi",
                device_name=f"SecuGen Scanner (WebAPI on port {port})",
                serial_number="WebAPI-Attached",
                resolution="300x400 @ 500 DPI",
                service_url=url,
                error_message=None,
            )
            self.last_status = status
            return status

        # 3. Not detected
        status = ScannerDeviceStatus(
            available=False,
            mode="not_connected",
            device_name="None",
            serial_number="N/A",
            resolution="N/A",
            service_url=None,
            error_message="SecuGen scanner or Bio Server service is not detected. Please verify USB connection or start sgibiosrv.exe.",
        )
        self.last_status = status
        return status

    def validate_image(self, image: np.ndarray, reported_quality: int) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate captured image.
        Returns:
            (is_valid: bool, error_msg: Optional[str], warning_msg: Optional[str])
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return False, "Captured image buffer is empty or corrupted.", None

        if image.ndim != 2:
            return False, f"Expected 2D grayscale image, received dimensions: {image.shape}", None

        h, w = image.shape
        if h < 100 or w < 100:
            return False, f"Invalid image dimensions: {w}x{h}. Minimum required is 100x100.", None

        # Check for blank / completely solid image
        std_dev = float(np.std(image))
        if std_dev < 3.0:
            return False, "Captured image appears blank or uniform. Please place finger firmly on the sensor.", None

        # Check mean intensity
        mean_val = float(np.mean(image))
        if mean_val < 5.0:
            return False, "Captured image is completely black. Sensor did not receive light.", None
        if mean_val > 250.0:
            return False, "Captured image is completely saturated white.", None

        # Check quality warning
        warning: Optional[str] = None
        if reported_quality > 0 and reported_quality < 40:
            warning = f"Scanner reported low image quality ({reported_quality}/100). Recapture is recommended if ridges are faint."
        elif std_dev < 15.0:
            warning = "Image contrast is relatively low. Please check ridge clarity before accepting."

        return True, None, warning

    def capture(self, timeout_sec: int = 20, quality_min: int = 30, allow_mock: bool = False) -> CaptureResult:
        """
        Trigger a physical capture from SecuGen scanner.
        Attempts Native SDK first; if unavailable, falls back to WebAPI.
        """
        # Try Native SDK first
        native_attempt_failed = False
        native_error = ""

        try:
            scanner = SecuGenNativeScanner()
            info = scanner.open()
            serial = info.DeviceSN.decode(errors="ignore").strip()
            w, h = int(info.ImageWidth), int(info.ImageHeight)

            # Capture image with optical sensor active
            image, quality = scanner.capture_image(timeout_sec=timeout_sec, quality_min=quality_min)
            scanner.close()

            is_valid, err, warn = self.validate_image(image, quality)
            if not is_valid:
                return CaptureResult(
                    success=False,
                    error_message=err,
                    scanner_model="SecuGen Hamster Pro 20",
                    serial_number=serial,
                    capture_method="Native Hardware Driver",
                )

            return CaptureResult(
                success=True,
                image=image,
                width=w,
                height=h,
                quality=quality,
                scanner_model="SecuGen Hamster Pro 20",
                serial_number=serial,
                capture_method="Native Hardware Driver",
                warning_message=warn,
            )

        except TimeoutError:
            return CaptureResult(
                success=False,
                error_message="Capture timed out: Sensor did not detect sufficient finger contact. Place finger flatly on the glass prism and apply steady pressure.",
                capture_method="Native Hardware Driver",
            )
        except Exception as e:
            native_attempt_failed = True
            native_error = str(e)

        # Fallback to WebAPI service
        webapi_info = self.check_webapi_port()
        if webapi_info:
            service_url, port = webapi_info
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

                license_str = os.getenv("SECUGEN_LIC", "")
                params = {
                    "Timeout": str(timeout_sec * 1000),
                    "Quality": str(quality_min),
                    "licstr": license_str,
                    "fakeDetection": "0",
                    "templateFormat": "0",
                    "imageWSQRate": "0.75",
                }

                data = urllib.parse.urlencode(params).encode("utf-8")
                req = urllib.request.Request(f"{service_url}{CAPTURE_ENDPOINT}", data=data)

                with urllib.request.urlopen(req, context=ctx, timeout=timeout_sec + 5.0) as resp:
                    content = resp.read().decode("utf-8")
                    result_json = json.loads(content)

                err_code = result_json.get("ErrorCode", -1)
                if err_code != 0:
                    err_msgs = {
                        1: "Scanner not found or disconnected. Check USB cable.",
                        2: "Capture timed out - no finger was detected on the scanner.",
                        3: "Poor image quality - sensor was unable to obtain a clear print.",
                        4: "License validation failed. Check SECUGEN_LIC in .env.",
                        10004: "Capture timed out - sensor did not detect finger contact.",
                    }
                    msg = err_msgs.get(err_code, f"SecuGen WebAPI returned ErrorCode {err_code}.")
                    return CaptureResult(
                        success=False,
                        error_message=msg,
                        capture_method=f"WebAPI (Port {port})",
                    )

                b64 = result_json.get("BMPBase64")
                if not b64:
                    return CaptureResult(
                        success=False,
                        error_message="Scanner returned response without image data.",
                        capture_method=f"WebAPI (Port {port})",
                    )

                raw_bytes = base64.b64decode(b64)
                image = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                quality = int(result_json.get("ImageQuality", 0) or 0)
                img_w = int(result_json.get("ImageWidth", 0) or (image.shape[1] if image is not None else 0))
                img_h = int(result_json.get("ImageHeight", 0) or (image.shape[0] if image is not None else 0))

                is_valid, err, warn = self.validate_image(image, quality)
                if not is_valid:
                    return CaptureResult(
                        success=False,
                        error_message=err,
                        scanner_model="SecuGen Hamster Pro 20",
                        serial_number="WebAPI-Device",
                        capture_method=f"WebAPI (Port {port})",
                    )

                return CaptureResult(
                    success=True,
                    image=image,
                    width=img_w,
                    height=img_h,
                    quality=quality,
                    scanner_model="SecuGen Hamster Pro 20",
                    serial_number="WebAPI-Device",
                    capture_method=f"WebAPI (Port {port})",
                    warning_message=warn,
                )

            except Exception as e:
                webapi_error = str(e)
                return CaptureResult(
                    success=False,
                    error_message=f"WebAPI capture failed: {webapi_error}",
                    capture_method=f"WebAPI (Port {port})",
                )

        # If allow_mock is explicitly enabled for offline testing
        if allow_mock:
            return self._generate_mock_capture()

        # Both failed and mock is disabled
        return CaptureResult(
            success=False,
            error_message=(
                f"SecuGen scanner unavailable: Native SDK error ({native_error or 'Failed'}), "
                "WebAPI service is not reachable on ports 8443 or 8000."
            ),
            capture_method="None",
        )

    def _generate_mock_capture(self) -> CaptureResult:
        """
        Generate a synthetic mock fingerprint for offline UI verification.
        Clearly watermarked so it is never confused with a real scan.
        """
        h, w = 400, 300
        # Create concentric ridges pattern
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        r = np.sqrt((x - center_x) ** 2 + 1.3 * (y - center_y) ** 2)
        pattern = (np.sin(r / 3.5) * 80 + 150).astype(np.uint8)

        # Add mock watermark text
        cv2.putText(pattern, "MOCK SIMULATION", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40,), 2)
        cv2.putText(pattern, "NO SCANNER DETECTED", (20, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,), 2)

        return CaptureResult(
            success=True,
            image=pattern,
            width=w,
            height=h,
            quality=50,
            scanner_model="MOCK_SIMULATOR (Offline Test)",
            serial_number="MOCK-00000",
            capture_method="Mock Simulator",
            warning_message="⚠️ Offline mock capture mode. Not a physical biometric scan.",
            is_mock=True,
        )

