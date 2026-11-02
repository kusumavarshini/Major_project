"""
FastAPI Backend Application for Fingerprint Distortion Rectification & Recognition.
Endpoints for 1:N identification, 1:1 verification, DDRNet rectification,
dataset statistics, and research evaluation analytics.
"""

from __future__ import annotations

import os
import sys
import time
import json
import socket
import base64
import binascii
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uuid
from mysql_auth import enroll_fingerprint
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from authentication import (
    identify_fingerprint,
    authenticate,
    classify_distortion,
    rectify_with_ddrnet,
    save_authentication_images,
    generate_diagnostic_overlay,
)
from matcher import MATCHER, SourceAFISMatcher
from database import (
    get_dataset_statistics,
    get_gallery_statistics,
    MATCH_THRESHOLD,
    test_mysql_connection,
)
from evaluation import (
    get_cached_metrics,
    run_evaluation_protocol,
)

app = FastAPI(
    title="Fingerprint Distortion Rectification & Recognition API",
    description="Professional biometric identification & research analytics dashboard API.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = CURRENT_DIR / "experiments" / "authentication_outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

FRONTEND_DIST_DIR = CURRENT_DIR / "frontend" / "dist"
if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")


# ============================================================
# HEALTH & INFO ENDPOINTS
# ============================================================

@app.get("/api/health")
def health_check():
    mysql_ok, mysql_msg = test_mysql_connection()
    return {
        "status": "healthy",
        "service": "Fingerprint Distortion Rectification & Recognition API",
        "mysql_connected": mysql_ok,
        "mysql_details": mysql_msg,
        "matcher": "SourceAFIS",
        "threshold": MATCH_THRESHOLD,
    }


@app.get("/api/dataset/stats")
@app.get("/dataset/stats")
def dataset_stats():
    """Retrieve actual dataset and template counts from MySQL database."""
    return get_dataset_statistics()


@app.get("/api/gallery/stats")
@app.get("/gallery/stats")
def gallery_stats():
    """
    Retrieve statistics from the operational gallery (fingersure_eval_db.fingerprint_gallery).
    This is the gallery used by the live identification pipeline.
    """
    return get_gallery_statistics()


@app.post("/api/fingerprint/enroll")
async def enroll_fingerprint_api(
    fingerprint: UploadFile = File(...),
    subject_id: str = Form("TEST-USER-001"),
):
    if not subject_id.strip():
        raise HTTPException(status_code=400, detail="Subject ID cannot be empty.")

    image_bytes = await fingerprint.read()
    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Invalid fingerprint image.")

    template = MATCHER.extract_template(image)

    if template.get("minutiae_count", 0) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough minutiae detected. Capture again.",
        )

    image_dir = CURRENT_DIR / "experiments" / "enrollment_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    filename = f"enrolled_{uuid.uuid4().hex}.png"
    image_path = image_dir / filename

    if not cv2.imwrite(str(image_path), image):
        raise HTTPException(status_code=500, detail="Could not save image.")

    try:
        fingerprint_id = enroll_fingerprint(
            subject_id=subject_id.strip(),
            filename=filename,
            image_path=str(image_path.resolve()),
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
            template=template,
        )
    except Exception as exc:
        image_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Fingerprint enrollment failed: {exc}",
        ) from exc

    return {
        "success": True,
        "message": "Fingerprint enrolled successfully.",
        "fingerprint_id": fingerprint_id,
        "subject_id": subject_id.strip(),
        "minutiae_count": template["minutiae_count"],
    }


# ============================================================
# SECUGEN HELPER FUNCTIONS
# ============================================================

def check_secugen_process_running() -> bool:
    """
    Check if SecuGen Bio Server process (sgibiosrv.exe) is actually running.
    This prevents false positives from other services using port 8000.
    """
    try:
        import psutil
        # Look for sgibiosrv.exe process
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] and 'sgibiosrv.exe' in proc.info['name'].lower():
                return True
        return False
    except ImportError:
        # If psutil not available, use subprocess as fallback on Windows
        try:
            import subprocess
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq sgibiosrv.exe'], 
                                  capture_output=True, text=True, timeout=3)
            return 'sgibiosrv.exe' in result.stdout
        except Exception:
            # If all process detection fails, return False to be safe
            return False


def check_port_connectivity(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a port is open and accepting connections using socket."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


async def validate_webapi_bioserver(host: str = 'localhost', port: int = 8443) -> bool:
    """
    Validate that port 8443 is actually SecuGen WebAPI Bio Server, not another service.
    Returns True if confirmed as SecuGen Bio Server, False otherwise.
    Enhanced validation to prevent false positives from other services on port 8443.
    """
    try:
        import httpx
    except ImportError:
        # If httpx not available, use process-based detection as fallback
        return check_secugen_process_running()
        
    try:
        # First check basic connectivity
        if not check_port_connectivity(host, port, timeout=1.0):
            return False
        
        # Additional validation: Check if SecuGen process is actually running
        if not check_secugen_process_running():
            return False
            
        # Attempt HTTPS request to verify service behavior - disable TLS verification for local service
        async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
            try:
                # Try root path - SecuGen Bio Server has specific timeout behavior
                response = await client.get(f'https://{host}:{port}/', timeout=2.0)
                # Any HTTP response indicates a web server, but need process confirmation above
                return response.status_code is not None
            except httpx.HTTPStatusError:
                # HTTP errors are still valid responses, means it's a web server
                # Combined with process check, this is likely SecuGen
                return True
            except (httpx.ConnectTimeout, httpx.ReadTimeout):
                # Timeouts are actually expected behavior for SecuGen Bio Server
                # If process is running and we get timeout, it's likely SecuGen
                return True
            except httpx.ConnectError:
                # Connection refused suggests not an HTTP server
                return False
                
    except Exception:
        # Fall back to process-based detection only
        return check_secugen_process_running()


async def check_connect_bridge_discovery(ports: List[int] = None) -> Optional[Dict[str, Any]]:
    """
    Check for SecuGen Connect Bridge using SGDISC /v1/discover endpoint.
    Returns discovery info if found, None if not available.
    """
    if ports is None:
        ports = [4494, 4495, 4496, 4497, 4498, 4499]
    
    try:
        import httpx
    except ImportError:
        # httpx not available, cannot check Connect Bridge
        return None
        
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            for port in ports:
                try:
                    response = await client.get(f'http://127.0.0.1:{port}/v1/discover')
                    if response.status_code == 200:
                        discovery_data = response.json()
                        return {
                            'port': port,
                            'discovery_data': discovery_data,
                            'endpoint': f'http://127.0.0.1:{port}'
                        }
                except Exception:
                    continue
        return None
    except Exception:
        return None


# ============================================================
# SECUGEN LIVE CAPTURE INTEGRATION
# ============================================================

# SecuGen Bio Server Configuration
SECUGEN_BASE_URL = "https://localhost:8443"
SECUGEN_CAPTURE_ENDPOINT = "/SGIFPCapture"

# Global flag to prevent concurrent capture requests
import threading
_capture_lock = threading.Lock()
_capture_in_progress = False

@app.get("/api/fingerprint/device-status")
@app.get("/fingerprint/device-status")
async def get_device_status():
    """Check SecuGen scanner availability (Native hardware driver or WebAPI service)."""
    # 1. Check Native Hardware driver first
    try:
        from secugen_scanner import SecuGenNativeScanner
        scanner = SecuGenNativeScanner()
        info = scanner.open()
        serial = info.DeviceSN.decode(errors='ignore').strip()
        scanner.close()
        return {
            "available": True,
            "scanner_type": "native_hardware",
            "device_name": f"SecuGen Hamster Pro 20 (SN: {serial})",
            "service_url": SECUGEN_BASE_URL,
            "capture_endpoint": SECUGEN_CAPTURE_ENDPOINT,
            "actionable_message": f"SecuGen Hamster Pro 20 ready (Hardware SN: {serial})",
            "error": None
        }
    except Exception:
        pass

    # 2. Fallback: Check HTTPS WebAPI service on port 8443
    https_available = check_port_connectivity('localhost', 8443)
    process_running = check_secugen_process_running()
    if https_available and process_running:
        return {
            "available": True,
            "scanner_type": "webapi_https",
            "device_name": "SecuGen Scanner (WebAPI HTTPS)",
            "service_url": SECUGEN_BASE_URL,
            "capture_endpoint": SECUGEN_CAPTURE_ENDPOINT,
            "actionable_message": "Scanner ready - HTTPS WebAPI on port 8443",
            "error": None
        }

    return {
        "available": False,
        "scanner_type": None,
        "device_name": None,
        "service_url": None,
        "actionable_message": "SecuGen Hamster Pro 20 USB scanner not detected. Please connect USB cable.",
        "error": "No SecuGen hardware or WebAPI service detected"
    }


@app.get("/api/fingerprint/secugen-config")
async def get_secugen_config():
    """
    Provide SecuGen configuration including license string for frontend capture.
    Issue 1 fix: Replace hardcoded empty license with environment variable.
    """
    # Read license string from environment (defaults to empty for evaluation)
    license_string = os.environ.get('SECUGEN_LIC', '')
    
    return {
        "licstr": license_string,
        "timeout": 10000,
        "quality": 50,
        "fake_detection": 0,
        "template_format": 0,
        "image_wsq_rate": 0.75
    }


_last_endpoint_check_time = 0.0
_last_endpoint_check_result = False

async def test_capture_endpoint_availability() -> bool:
    """
    Test actual SecuGen WebAPI capture endpoint functionality.
    Cached for 60 seconds to prevent continuous optical scanner bus contention.
    """
    global _last_endpoint_check_time, _last_endpoint_check_result
    now = time.time()
    if (now - _last_endpoint_check_time) < 60.0:
        return _last_endpoint_check_result

    try:
        import httpx
        
        async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
            test_params = {
                'Timeout': '1000',
                'Quality': '50',
                'licstr': '',
                'fakeDetection': '0',
                'templateFormat': '0',
                'imageWSQRate': '0.75'
            }
            
            try:
                response = await client.post(
                    f"{SECUGEN_BASE_URL}{SECUGEN_CAPTURE_ENDPOINT}",
                    data=test_params,
                    timeout=2.0
                )
                
                if response.status_code == 200:
                    try:
                        json_response = response.json()
                        if 'ErrorCode' in json_response:
                            _last_endpoint_check_time = now
                            _last_endpoint_check_result = True
                            return True
                        _last_endpoint_check_result = False
                        return False
                    except:
                        _last_endpoint_check_result = False
                        return False
                        
                _last_endpoint_check_result = False
                return False
                
            except (httpx.ConnectError, httpx.TimeoutException):
                _last_endpoint_check_result = False
                return False
                
    except ImportError:
        # If httpx not available, assume endpoint works if port is open
        # (this is the fallback behavior that preserves existing functionality)
        return True
    except Exception:
        return False


@app.post("/api/fingerprint/scan-live")
@app.post("/fingerprint/scan-live")
async def scan_live_fingerprint(
    top_n: int = Form(5, description="Number of top candidates to return"),
    threshold: Optional[float] = Form(None, description="Custom matching threshold"),
    force_rectification: bool = Form(False, description="Force DDRNet rectification")
):
    """
    Capture live optical fingerprint from SecuGen Hamster Pro 20 directly on server
    and run through 1:N identification pipeline.
    """
    global _capture_in_progress

    with _capture_lock:
        if _capture_in_progress:
            raise HTTPException(
                status_code=409,
                detail="Another fingerprint scan is currently in progress. Please wait."
            )
        _capture_in_progress = True

    try:
        captured_image = None
        quality = 0
        method = "Native Hardware"

        # Try Native Driver first
        try:
            from secugen_scanner import SecuGenNativeScanner
            with SecuGenNativeScanner() as scanner:
                captured_image, quality = scanner.capture_image(timeout_sec=20, quality_min=30)
                method = "SecuGen Native FDx Driver"
        except TimeoutError:
            raise HTTPException(
                status_code=408,
                detail="Capture timed out: Please place your finger firmly and flatly on the scanner prism."
            )
        except Exception as native_err:
            # Fallback to WebAPI via Python
            try:
                import urllib.request, urllib.parse, ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                params = {
                    "Timeout": "20000",
                    "Quality": "40",
                    "licstr": os.getenv("SECUGEN_LIC", ""),
                    "fakeDetection": "0",
                    "templateFormat": "0",
                    "imageWSQRate": "0.75",
                }
                data = urllib.parse.urlencode(params).encode("utf-8")
                req = urllib.request.Request(f"{SECUGEN_BASE_URL}{SECUGEN_CAPTURE_ENDPOINT}", data=data)
                with urllib.request.urlopen(req, context=ctx, timeout=25.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    if resp_data.get("ErrorCode", -1) != 0:
                        raise ValueError(f"SecuGen ErrorCode: {resp_data.get('ErrorCode')}")
                    b64 = resp_data.get("BMPBase64")
                    raw = base64.b64decode(b64)
                    captured_image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    quality = resp_data.get("ImageQuality", 0)
                    method = "SecuGen WebAPI Fallback"
            except Exception as webapi_err:
                raise HTTPException(
                    status_code=500,
                    detail=f"Scanner capture failed: Native driver error ({native_err}), WebAPI error ({webapi_err})"
                )

        if captured_image is None or captured_image.size == 0:
            raise HTTPException(status_code=500, detail="Captured image is empty or invalid.")

        if threshold is not None:
            t_thresh = threshold if threshold > 1.0 else (threshold * 100.0 if threshold > 0.0 else MATCH_THRESHOLD)
        else:
            t_thresh = MATCH_THRESHOLD

        # Run identification pipeline
        result = identify_fingerprint(
            fingerprint_image=captured_image,
            top_n=top_n,
            threshold=t_thresh,
            force_rectification=force_rectification
        )

        h, w = captured_image.shape
        result['capture_method'] = method
        result['capture_dimensions'] = f"{w}x{h}"
        result['capture_quality'] = quality

        return result

    finally:
        with _capture_lock:
            _capture_in_progress = False


@app.post("/api/fingerprint/capture-and-search")
@app.post("/fingerprint/capture-and-search")
@app.post("/api/fingerprint/capture-from-bmp")
@app.post("/fingerprint/capture-from-bmp")
async def capture_from_bmp_base64(
    bmp_base64: str = Form(..., description="Base64-encoded BMP image from SecuGen WebAPI"),
    image_width: int = Form(..., description="Image width from SecuGen capture"),
    image_height: int = Form(..., description="Image height from SecuGen capture"),
    image_quality: int = Form(..., description="Image quality from SecuGen capture"),
    top_n: int = Form(5, description="Number of top candidates to return"),
    threshold: Optional[float] = Form(None, description="Custom matching threshold"),
    force_rectification: bool = Form(False, description="Force DDRNet rectification")
):
    """
    Process fingerprint captured via SecuGen WebAPI JavaScript and perform identification.
    Expects BMPBase64 data from SGIFPCapture endpoint response.
    """
    global _capture_in_progress
    
    # Prevent concurrent capture requests with thread-safe locking
    with _capture_lock:
        if _capture_in_progress:
            raise HTTPException(
                status_code=409,
                detail="Another capture request is already in progress. Please wait."
            )
        _capture_in_progress = True
    
    try:
        if threshold is not None:
            t_thresh = threshold if threshold > 1.0 else (threshold * 100.0 if threshold > 0.0 else MATCH_THRESHOLD)
        else:
            t_thresh = MATCH_THRESHOLD
        
        # Validate and decode BMP Base64 data
        try:
            # Remove data URL prefix if present
            if bmp_base64.startswith('data:image/bmp;base64,'):
                bmp_base64 = bmp_base64.split(',')[1]
            
            # Validate base64 format and length
            if not bmp_base64 or len(bmp_base64) < 100:
                raise ValueError("BMP Base64 data too short or empty")
                
            # Decode base64 to bytes
            image_bytes = base64.b64decode(bmp_base64, validate=True)
            
            # Validate decoded data length
            if len(image_bytes) < 1000:  # BMP header + minimal image data
                raise ValueError("Decoded BMP data too small")
            
            # Decode BMP image using OpenCV
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            captured_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            
            # Validate decoded image
            if captured_image is None or captured_image.size == 0:
                raise ValueError("Failed to decode BMP image data")
            
            # Validate image dimensions match reported values
            actual_height, actual_width = captured_image.shape
            if abs(actual_width - image_width) > 10 or abs(actual_height - image_height) > 10:
                print(f"Warning: Image dimension mismatch. Reported: {image_width}x{image_height}, Actual: {actual_width}x{actual_height}")
                
        except (base64.binascii.Error, ValueError, cv2.error) as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid BMP Base64 data",
                    "actionable_message": f"Failed to decode captured fingerprint image: {str(e)}",
                    "possible_causes": [
                        "Corrupted capture data from SecuGen WebAPI",
                        "Invalid Base64 encoding",
                        "Unsupported BMP format"
                    ]
                }
            )
        
        # Run the existing identification pipeline (unchanged)
        result = identify_fingerprint(
            fingerprint_image=captured_image,
            top_n=top_n,
            threshold=t_thresh,
            force_rectification=force_rectification
        )
        
        # Add capture metadata to result
        result['capture_method'] = 'SecuGen WebAPI JavaScript'
        result['capture_dimensions'] = f"{image_width}x{image_height}"
        result['capture_quality'] = image_quality
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Capture processing error: {str(e)}"
        )
    finally:
        with _capture_lock:
            _capture_in_progress = False


async def capture_via_connect_bridge(bridge_info: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Capture fingerprint using SecuGen Connect Bridge /v1/fingerprint/capture endpoint.
    """
    try:
        import httpx
        import base64
    except ImportError as e:
        # Required dependencies not available
        raise Exception(f"Missing required dependency: {e}")
        
    try:
        endpoint_url = f"{bridge_info['endpoint']}/v1/fingerprint/capture"
        
        capture_payload = {
            "timeout": 10000,
            "format": "png",
            "quality": "high"
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                endpoint_url,
                json=capture_payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                capture_data = response.json()
                
                if (capture_data.get('status') == 'success' and 
                    capture_data.get('image')):
                    
                    # Extract base64 image data with validation
                    image_data_str = capture_data['image']
                    if ',' in image_data_str:
                        image_data_str = image_data_str.split(',')[1]
                    
                    try:
                        # Comprehensive base64 validation and decoding
                        
                        # 1. Basic format validation
                        if not image_data_str or len(image_data_str) < 4:
                            return None
                            
                        # 2. Check for valid base64 characters
                        if not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in image_data_str):
                            return None
                        
                        # 3. Validate base64 format and decode
                        image_bytes = base64.b64decode(image_data_str, validate=True)
                        
                        # 4. Check decoded data length (minimum reasonable image size)
                        if len(image_bytes) < 100:  # Too small to be a valid image
                            return None
                            
                        # 5. Attempt image decoding and validate result
                        image = cv2.imdecode(
                            np.frombuffer(image_bytes, dtype=np.uint8),
                            cv2.IMREAD_GRAYSCALE
                        )
                        
                        # 6. Validate decoded image properties
                        if image is not None and image.size > 0:
                            # Check for reasonable image dimensions
                            height, width = image.shape
                            if 50 <= width <= 2000 and 50 <= height <= 2000:
                                return image
                            
                    except (base64.binascii.Error, ValueError, cv2.error) as e:
                        # Handle malformed base64, invalid image data, or CV2 decode errors
                        return None
            
            return None
            
    except Exception:
        return None


# ============================================================
# PRIMARY USER WORKFLOW: 1:N IDENTIFICATION  
# ============================================================

@app.post("/fingerprint/search")
@app.post("/api/fingerprint/search")
async def fingerprint_search(
    fingerprint: UploadFile = File(..., description="Query fingerprint image"),
    top_n: int = Form(5, description="Number of top candidates to return"),
    threshold: Optional[float] = Form(None, description="Custom matching threshold"),
    force_rectification: bool = Form(False, description="Force DDRNet rectification"),
):
    """
    1:N Biometric Identification: handles preprocessing, DDRNet rectification,
    SourceAFIS template generation, and 1:N database search.
    """
    try:
        image_bytes = await fingerprint.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")

        if threshold is not None:
            t_thresh = threshold if threshold > 1.0 else (threshold * 100.0 if threshold > 0.0 else MATCH_THRESHOLD)
        else:
            t_thresh = MATCH_THRESHOLD

        result = identify_fingerprint(
            fingerprint_image=image,
            top_n=top_n,
            threshold=t_thresh,
            force_rectification=force_rectification,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"1:N Search Error: {str(e)}")


# ============================================================
# 1:1 BIOMETRIC VERIFICATION
# ============================================================

@app.post("/fingerprint/match")
@app.post("/api/fingerprint/match")
async def fingerprint_match(
    probe_fingerprint: UploadFile = File(...),
    reference_fingerprint: Optional[UploadFile] = File(None),
    subject_id: Optional[str] = Form(None),
    threshold: Optional[float] = Form(None),
):
    """
    1:1 Biometric Verification between two fingerprints or a probe and enrolled subject.
    """
    try:
        if threshold is not None:
            t_thresh = threshold if threshold > 1.0 else (threshold * 100.0 if threshold > 0.0 else MATCH_THRESHOLD)
        else:
            t_thresh = MATCH_THRESHOLD

        probe_bytes = await probe_fingerprint.read()
        probe_img = cv2.imdecode(np.frombuffer(probe_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        if probe_img is None:
            raise HTTPException(status_code=400, detail="Invalid probe image file.")

        if reference_fingerprint is not None:
            ref_bytes = await reference_fingerprint.read()
            ref_img = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                raise HTTPException(status_code=400, detail="Invalid reference image file.")

            tpl_probe = MATCHER.extract_template(probe_img)
            tpl_ref = MATCHER.extract_template(ref_img)
            res = MATCHER.match_1to1(tpl_probe, tpl_ref, threshold=t_thresh)

            return {
                "matcher": "SourceAFIS",
                "score": res.score,
                "raw_score": res.raw_score,
                "match": res.match,
                "matched_minutiae": res.matched_minutiae_count,
                "runtime_ms": res.runtime_ms,
                "threshold": t_thresh,
            }

        if not subject_id:
            raise HTTPException(
                status_code=400,
                detail="Either reference_fingerprint or subject_id must be provided.",
            )

        auth_res = authenticate(identity_code=subject_id, fingerprint_image=probe_img)
        return {
            "matcher": "SourceAFIS",
            "subject_id": subject_id,
            "best_score": auth_res["best_score"],
            "best_raw_score": auth_res["best_raw_score"],
            "match": auth_res["authenticated"],
            "threshold": t_thresh,
            "scores": auth_res["scores"],
            "original_image_url": f"/outputs/{os.path.basename(auth_res['saved_original_path'])}" if auth_res.get("saved_original_path") else None,
            "rectified_image_url": f"/outputs/{os.path.basename(auth_res['saved_rectified_path'])}" if auth_res.get("saved_rectified_path") else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RECTIFICATION ENDPOINT
# ============================================================

@app.post("/fingerprint/rectify")
@app.post("/api/fingerprint/rectify")
async def fingerprint_rectify(
    fingerprint: UploadFile = File(...),
):
    """Run DDRNet geometric rectification on uploaded fingerprint image."""
    try:
        image_bytes = await fingerprint.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        cls_res = classify_distortion(image)
        rectified = rectify_with_ddrnet(image)

        saved_orig, saved_rect = save_authentication_images(
            original_image=image,
            rectified_image=rectified,
            prefix="rectify",
        )

        return {
            "is_distorted": cls_res["is_distorted"],
            "distortion_probability": cls_res["prob_distorted"],
            "clean_probability": cls_res["prob_clean"],
            "distorted_probability": cls_res["prob_distorted"],
            "predicted_class": cls_res["predicted_class"],
            "original_image_url": f"/outputs/{os.path.basename(saved_orig)}",
            "rectified_image_url": f"/outputs/{os.path.basename(saved_rect)}",
            "image_width": int(rectified.shape[1]),
            "image_height": int(rectified.shape[0]),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# DIAGNOSTIC INSPECTION ENDPOINT
# ============================================================

@app.post("/fingerprint/diagnostic/inspect")
@app.post("/api/fingerprint/diagnostic/inspect")
async def fingerprint_diagnostic_inspect(
    fingerprint: UploadFile = File(..., description="Query image to visually inspect"),
):
    """
    Diagnostic: original image, preprocessed image, DDRNet rectified image,
    and extracted minutiae overlay.
    """
    try:
        image_bytes = await fingerprint.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")

        diagnostic_data = generate_diagnostic_overlay(image)
        return diagnostic_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {str(e)}")


# ============================================================
# EVALUATION & RESEARCH ANALYTICS ENDPOINTS
# ============================================================

@app.get("/evaluation/metrics")
@app.get("/api/evaluation/metrics")
def get_metrics():
    """Retrieve calculated biometric metrics (ROC-AUC, EER, FAR, FRR)."""
    return get_cached_metrics()


@app.get("/evaluation/roc")
@app.get("/api/evaluation/roc")
def get_roc_data():
    """Retrieve ROC curve coordinates and score distributions."""
    metrics = get_cached_metrics()
    if metrics.get("status") == "Evaluation not run":
        return {"status": "Evaluation not run"}

    return {
        "roc_curve": metrics.get("roc_curve", {}),
        "score_distribution": metrics.get("score_distribution", {}),
        "rectified_metrics": metrics.get("rectified_metrics", {}),
        "original_metrics": metrics.get("original_metrics", {}),
    }


@app.post("/evaluation/run")
@app.post("/api/evaluation/run")
async def run_evaluation(
    sample_limit: Optional[int] = Form(None),
):
    """
    Run evaluation protocol comparing original distorted vs DDRNet rectified recognition.
    """
    try:
        res = run_evaluation_protocol(sample_limit=sample_limit)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation protocol failed: {str(e)}")


# ============================================================
# TOP-1 & TOP-5 ACCURACY EVALUATION ENDPOINTS
# ============================================================

_accuracy_eval_lock = threading.Lock()
_accuracy_eval_in_progress = False


@app.get("/evaluation/accuracy")
@app.get("/api/evaluation/accuracy")
def get_matching_accuracy():
    """Retrieve measured Top-1 and Top-5 fingerprint matching accuracy metrics."""
    summary_file = CURRENT_DIR / "experiments" / "evaluation_results" / "matching_accuracy_summary.json"
    if not summary_file.exists():
        # Fall back to regenerating or checking cache
        from evaluate_matching_accuracy import run_accuracy_evaluation, EXISTING_EVAL_CSV
        if EXISTING_EVAL_CSV.exists():
            try:
                return run_accuracy_evaluation(reuse_cached_predictions=True)
            except Exception:
                pass
        return {
            "status": "Evaluation not run",
            "message": "Evaluation has not been executed yet. Run evaluate_matching_accuracy.py or trigger via /api/evaluation/accuracy/run"
        }

    try:
        with open(summary_file, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation summary: {str(e)}")


@app.post("/evaluation/accuracy/run")
@app.post("/api/evaluation/accuracy/run")
async def run_matching_accuracy_api(
    sample_limit: Optional[int] = Form(None, description="Limit number of queries evaluated for speed"),
    force_rectification: bool = Form(False, description="Force DDRNet rectification on all queries"),
    reuse_cached_predictions: bool = Form(False, description="Reuse verified precomputed 317-query predictions"),
):
    """
    Execute Top-1 and Top-5 matching accuracy evaluation on the gallery.
    """
    global _accuracy_eval_in_progress
    with _accuracy_eval_lock:
        if _accuracy_eval_in_progress:
            raise HTTPException(
                status_code=409,
                detail="Accuracy evaluation is already running. Please wait for the current job to finish."
            )
        _accuracy_eval_in_progress = True

    try:
        from evaluate_matching_accuracy import run_accuracy_evaluation
        summary = run_accuracy_evaluation(
            sample_limit=sample_limit,
            force_rectification=force_rectification,
            reuse_cached_predictions=reuse_cached_predictions,
        )
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    finally:
        with _accuracy_eval_lock:
            _accuracy_eval_in_progress = False


@app.get("/evaluation/accuracy/predictions")
@app.get("/api/evaluation/accuracy/predictions")
def get_matching_predictions(limit: int = 50, offset: int = 0):
    """Retrieve per-query prediction records for audit and dashboard inspection."""
    import csv
    pred_file = CURRENT_DIR / "experiments" / "evaluation_results" / "matching_accuracy_predictions.csv"
    if not pred_file.exists():
        return {"total": 0, "predictions": []}

    rows = []
    try:
        with open(pred_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return {
            "total": len(rows),
            "limit": limit,
            "offset": offset,
            "predictions": rows[offset : offset + limit],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read predictions: {str(e)}")



# ============================================================
# FRONTEND SPA SERVING
# ============================================================

@app.get("/")
def serve_index():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Fingerprint Recognition API is active. Frontend build not found at frontend/dist."}


@app.get("/app")
def serve_app():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return FileResponse("frontend/index.html")


# ============================================================
# MAIN STARTUP SECTION
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting FastAPI Backend Server...")
    print("SecuGen WebAPI should be running on port 8000")
    print("FastAPI Backend will start on port 8001")
    print("Frontend dev server available at http://localhost:5173")
    print()
    
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )