"""
FastAPI Backend Application for Fingerprint Distortion Rectification & Recognition.
Endpoints for 1:N identification, 1:1 verification, DDRNet rectification,
dataset statistics, and research evaluation analytics.
"""

from __future__ import annotations

import os
import sys
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
    Retrieve statistics from the operational FVC gallery (newdb.external_fvc_clean_gallery).
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
# SECUGEN LIVE CAPTURE INTEGRATION
# ============================================================

@app.get("/api/fingerprint/device-status")
@app.get("/fingerprint/device-status")
async def get_device_status():
    """Check SecuGen Bio Server availability and device status."""
    try:
        import httpx
        
        # Try to connect to SecuGen Bio Server
        async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
            try:
                # Try HTTPS first
                response = await client.get('https://localhost:8000/status')
                if response.status_code == 200:
                    return {
                        "available": True,
                        "device_name": "SecuGen Hamster Pro 20",
                        "error": None,
                        "service_url": "https://localhost:8000"
                    }
            except Exception:
                # Fall back to HTTP
                response = await client.get('http://localhost:8000/status')
                if response.status_code == 200:
                    return {
                        "available": True,
                        "device_name": "SecuGen Hamster Pro 20", 
                        "error": None,
                        "service_url": "http://localhost:8000"
                    }
                    
        return {
            "available": False,
            "device_name": None,
            "error": "SecuGen Bio Server not responding on port 8000"
        }
        
    except Exception as e:
        return {
            "available": False,
            "device_name": None,
            "error": f"Device status check failed: {str(e)}"
        }


@app.post("/api/fingerprint/capture-and-search")
@app.post("/fingerprint/capture-and-search")
async def capture_and_search_fingerprint(
    top_n: int = Form(5, description="Number of top candidates to return"),
    threshold: Optional[float] = Form(None, description="Custom matching threshold"),
    force_rectification: bool = Form(False, description="Force DDRNet rectification")
):
    """
    Capture fingerprint via SecuGen Bio Server and immediately search gallery.
    This endpoint acts as a server-side proxy to avoid browser CORS issues.
    """
    try:
        import httpx
        import base64
        
        if threshold is not None:
            t_thresh = threshold if threshold > 1.0 else (threshold * 100.0 if threshold > 0.0 else MATCH_THRESHOLD)
        else:
            t_thresh = MATCH_THRESHOLD
        
        # Try multiple SecuGen Bio Server endpoints in order
        capture_endpoints = [
            {
                'url': 'https://localhost:8000/scan',
                'method': 'POST',
                'body': {'timeout': 10000, 'format': 'base64', 'quality': 'high'}
            },
            {
                'url': 'https://localhost:8000/capture', 
                'method': 'POST',
                'body': {'timeout': 10000, 'captureMode': 'single'}
            },
            {
                'url': 'https://localhost:8000/FPCapture',
                'method': 'POST', 
                'body': {'timeout': 10000}
            },
            {
                'url': 'http://localhost:8000/scan',
                'method': 'POST',
                'body': {'timeout': 10000, 'format': 'base64'}
            }
        ]
        
        captured_image = None
        successful_endpoint = None
        
        async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
            for endpoint in capture_endpoints:
                try:
                    response = await client.request(
                        method=endpoint['method'],
                        url=endpoint['url'],
                        json=endpoint['body'],
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        capture_data = response.json()
                        
                        # Check for successful capture response
                        if (capture_data.get('status') == 'success' and 
                            capture_data.get('image')):
                            
                            # Extract base64 image data
                            image_data_str = capture_data['image']
                            if ',' in image_data_str:
                                image_data_str = image_data_str.split(',')[1]
                            
                            # Decode base64 to numpy array
                            image_bytes = base64.b64decode(image_data_str)
                            image = cv2.imdecode(
                                np.frombuffer(image_bytes, dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE
                            )
                            
                            if image is not None and image.size > 0:
                                captured_image = image
                                successful_endpoint = endpoint['url']
                                break
                                
                except Exception as e:
                    # Continue to next endpoint if this one fails
                    continue
        
        if captured_image is None:
            raise HTTPException(
                status_code=502,
                detail="Failed to capture fingerprint from SecuGen Bio Server. Please ensure the service is running and a finger is placed on the scanner."
            )
        
        # Run the existing identification pipeline
        result = identify_fingerprint(
            fingerprint_image=captured_image,
            top_n=top_n,
            threshold=t_thresh,
            force_rectification=force_rectification
        )
        
        # Add capture metadata to result
        result['capture_method'] = 'SecuGen Bio Server'
        result['capture_endpoint'] = successful_endpoint
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Capture and search error: {str(e)}"
        )


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
