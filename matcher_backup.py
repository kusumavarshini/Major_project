"""
Fingerprint Matcher Abstraction & SourceAFIS Implementation.
Completely replaces legacy Bozorth3 matching pipeline with pure SourceAFIS.
"""

from __future__ import annotations

import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# SourceAFIS underlying algorithmic stack from afis package
import afis.algorithms.segmentation as seg
import afis.algorithms.orientation as ori
import afis.algorithms.frequency as freq
import afis.algorithms.enhancement as enh
import afis.algorithms.skeleton as skel
import afis.algorithms.minutiae as minu
import afis.algorithms.safis as safis

from database import MATCH_THRESHOLD


@dataclass
class MatchResult:
    score: float
    raw_score: float
    match: bool
    matched_minutiae_count: int
    runtime_ms: float
    threshold: float
    pairs: list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "raw_score": round(self.raw_score, 2),
            "match": self.match,
            "matched_minutiae_count": self.matched_minutiae_count,
            "runtime_ms": round(self.runtime_ms, 2),
            "threshold": self.threshold,
        }


class FingerprintMatcher(ABC):
    """Abstract base class for biometric fingerprint matching."""

    @abstractmethod
    def extract_template(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract template representation from single-channel grayscale image."""
        pass

    @abstractmethod
    def match_1to1(
        self,
        probe_template: Dict[str, Any],
        candidate_template: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """Perform 1:1 biometric verification between probe and candidate templates."""
        pass

    @abstractmethod
    def identify_1toN(
        self,
        probe_template: Dict[str, Any],
        gallery: List[Dict[str, Any]],
        top_n: int = 5,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform 1:N biometric identification against gallery of enrolled templates."""
        pass


class SourceAFISMatcher(FingerprintMatcher):
    """
    SourceAFIS-compliant matcher using rotation/translation-invariant edge pairing.
    Does NOT depend on MINDTCT or Bozorth3.
    """

    def __init__(
        self,
        default_threshold: float = MATCH_THRESHOLD,
        block_size: int = 16,
        max_minutiae: int = 80,
    ):
        self.default_threshold = default_threshold
        self.block_size = block_size
        self.max_minutiae = max_minutiae
        self.params = safis.SafisParams(
            max_neighbors=8,
            max_edge_length=300.0,
            length_tol=8.0,
            angle_tol=0.35,
            max_roots=15,
        )

    def extract_template(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract minutiae and invariant structural representation.
        Operates directly on grayscale fingerprint image.
        Standardizes biometric scale to (384, 384) so minutiae pixel distances
        and edge lengths strictly match across all query and gallery templates.
        """
        if image is None or image.size == 0:
            raise ValueError("Cannot extract template from empty image.")

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        orig_h, orig_w = image.shape
        # Standardize biometric processing resolution to exactly 384x384
        # Ensures all edge lengths, distances, and ridge frequencies match across probes and gallery
        TARGET_BIOMETRIC_SIZE = (384, 384)
        if (orig_w, orig_h) != TARGET_BIOMETRIC_SIZE:
            proc_img = cv2.resize(image, TARGET_BIOMETRIC_SIZE, interpolation=cv2.INTER_CUBIC)
        else:
            proc_img = image

        t0 = time.time()

        # 1. Normalization
        normed = (proc_img.astype(np.float64) - float(np.mean(proc_img))) / (float(np.std(proc_img)) + 1e-6)

        # 2. Segmentation & Orientation
        mask, _ = seg.segment(normed, block_size=self.block_size)
        orientation = ori.estimate_orientation(normed, block_size=self.block_size)

        # 3. Frequency & Gabor enhancement
        frequency = freq.estimate_frequency(normed, orientation, block_size=self.block_size)
        frequency = freq.fill_invalid(frequency, mask)
        enhanced = enh.enhance(
            normed,
            orientation,
            frequency,
            mask=mask,
            block_size=self.block_size,
            angle_steps=8,
        )

        # 4. Binarization & Skeletonization
        binary = skel.binarize(enhanced, mask=mask, block_size=self.block_size)
        skeleton = skel.skeletonize(binary)
        skeleton = skel.clean_skeleton(skeleton)

        # 5. Minutiae extraction and pruning
        raw_minutiae = minu.extract_minutiae(skeleton, orientation, mask=mask, border_margin=12)
        pruned = minu.prune_false_minutiae(
            raw_minutiae,
            skeleton,
            mask=mask,
            min_distance=8.0,
            max_minutiae=self.max_minutiae,
        )
        minutiae_arr = minu.minutiae_to_array(pruned)
        extract_ms = (time.time() - t0) * 1000.0

        return {
            "version": "sourceafis_v1",
            "minutiae": minutiae_arr,
            "minutiae_count": int(minutiae_arr.shape[0]),
            "image_width": int(orig_w),
            "image_height": int(orig_h),
            "extraction_time_ms": round(extract_ms, 2),
        }

    def match_1to1(
        self,
        probe_template: Dict[str, Any],
        candidate_template: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """Match two SourceAFIS templates."""
        t_thresh = self.default_threshold if threshold is None else threshold
        t0 = time.time()

        minu_a = probe_template.get("minutiae")
        minu_b = candidate_template.get("minutiae")

        if minu_a is None or minu_b is None or len(minu_a) < 2 or len(minu_b) < 2:
            runtime_ms = (time.time() - t0) * 1000.0
            return MatchResult(
                score=0.0,
                raw_score=0.0,
                match=False,
                matched_minutiae_count=0,
                runtime_ms=runtime_ms,
                threshold=t_thresh,
                pairs=[],
            )

        score, raw, pairs = safis.match_minutiae(minu_a, minu_b, params=self.params)
        runtime_ms = (time.time() - t0) * 1000.0

        raw_val = float(raw)
        decision = bool(raw_val >= t_thresh)

        return MatchResult(
            score=round(raw_val, 2),
            raw_score=round(raw_val, 2),
            match=decision,
            matched_minutiae_count=len(pairs),
            runtime_ms=runtime_ms,
            threshold=t_thresh,
            pairs=pairs,
        )

    def identify_1toN(
        self,
        probe_template: Dict[str, Any],
        gallery: List[Dict[str, Any]],
        top_n: int = 5,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fast 1:N Biometric Identification using the official SourceAFIS pattern:
        1. Indexes the probe template edges ONCE.
        2. Caches candidate invariant edges and quantized length buckets in memory.
        3. Efficiently matches each candidate via root-edge lookup and tree-crawl.
        4. Excludes failed/empty candidates (minutiae < 2).
        5. Returns candidates sorted descending by raw SourceAFIS score.
        """
        t_thresh = self.default_threshold if threshold is None else threshold
        t0 = time.time()

        if not gallery:
            return {
                "matcher": "SourceAFIS",
                "query_minutiae_count": probe_template.get("minutiae_count", 0),
                "best_match": None,
                "candidates": [],
                "threshold": t_thresh,
                "processing_time_ms": 0.0,
                "total_searched": 0,
            }

        probe_minu = probe_template.get("minutiae")
        if probe_minu is None or len(probe_minu) < 2:
            return {
                "matcher": "SourceAFIS",
                "query_minutiae_count": 0,
                "best_match": None,
                "candidates": [],
                "threshold": t_thresh,
                "processing_time_ms": round((time.time() - t0) * 1000.0, 2),
                "total_searched": len(gallery),
            }

        # 1. Build probe invariant edges ONCE
        ea = safis.build_edges(probe_minu, self.params)
        length_tol = float(self.params.length_tol) or 1e-6
        angle_tol = self.params.angle_tol
        max_roots = self.params.max_roots

        candidate_results = []

        # 2. Sequential candidate matching over pre-indexed memory gallery
        for cand_record in gallery:
            cand_tpl = cand_record.get("template", cand_record)
            
            # Ensure candidate edges & buckets are cached
            cand_edges = cand_tpl.get("_edges")
            cand_buckets = cand_tpl.get("_buckets")
            if cand_edges is None or cand_buckets is None:
                c_minu = cand_tpl.get("minutiae")
                if c_minu is not None and len(c_minu) >= 2:
                    cand_edges = safis.build_edges(c_minu, self.params)
                    cand_buckets = {}
                    for e in cand_edges:
                        k = int(round(e.length / length_tol))
                        cand_buckets.setdefault(k, []).append(e)
                    cand_tpl["_edges"] = cand_edges
                    cand_tpl["_buckets"] = cand_buckets
                else:
                    cand_edges = []
                    cand_buckets = {}
                    cand_tpl["_edges"] = cand_edges
                    cand_tpl["_buckets"] = cand_buckets

            if not cand_edges or not ea:
                raw_score = 0.0
                pairs_count = 0
            else:
                # Find compatible root edges
                roots = []
                for pe in ea:
                    key = int(round(pe.length / length_tol))
                    for k in (key - 1, key, key + 1):
                        b = cand_buckets.get(k)
                        if not b:
                            continue
                        for ce in b:
                            if safis._edges_compatible(pe, ce, self.params):
                                close = 1.0 - (
                                    abs(pe.length - ce.length) / length_tol
                                    + safis._angle_diff(pe.ref_angle, ce.ref_angle) / angle_tol
                                    + safis._angle_diff(pe.neighbor_angle, ce.neighbor_angle) / angle_tol
                                ) / 3.0
                                roots.append((close, pe, ce))

                if not roots:
                    raw_score = 0.0
                    pairs_count = 0
                else:
                    roots.sort(key=lambda t: t[0], reverse=True)
                    roots = roots[:max_roots]
                    best_n = 0
                    best_pairs_cnt = 0
                    for _c, pe, ce in roots:
                        pairing, pairs = safis._crawl(ea, cand_edges, pe.ref, ce.ref, pe.neighbor, ce.neighbor, self.params)
                        if len(pairing) > best_n:
                            best_n = len(pairing)
                            best_pairs_cnt = len(pairs)
                    raw_score = float(best_n) * 5.0
                    if best_n >= 2:
                        raw_score += float(best_pairs_cnt) * 1.5
                    pairs_count = best_pairs_cnt

            is_match = bool(raw_score >= t_thresh)
            candidate_results.append({
                "fingerprint_id": cand_record.get("fingerprint_id"),
                "subject_id": cand_record.get("subject_id"),
                "family_id": cand_record.get("family_id"),
                "member_type": cand_record.get("member_type"),
                "filename": cand_record.get("filename"),
                "image_path": cand_record.get("image_path"),
                "score": round(raw_score, 2),
                "raw_score": round(raw_score, 2),
                "matched_minutiae": pairs_count,
                "match": is_match,
            })

        # Sort descending by raw_score, then matched_minutiae
        candidate_results.sort(key=lambda x: (x["raw_score"], x["matched_minutiae"]), reverse=True)

        top_candidates = candidate_results[:top_n]
        best_candidate = top_candidates[0] if top_candidates else None

        total_runtime_ms = (time.time() - t0) * 1000.0

        return {
            "matcher": "SourceAFIS",
            "query_minutiae_count": probe_template.get("minutiae_count", 0),
            "best_match": best_candidate,
            "candidates": top_candidates,
            "threshold": t_thresh,
            "processing_time_ms": round(total_runtime_ms, 2),
            "total_searched": len(gallery),
        }


# Global singleton instance for high performance
MATCHER = SourceAFISMatcher()
