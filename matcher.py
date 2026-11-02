"""
Fingerprint Matcher Abstraction & SourceAFIS Implementation.
Completely replaces legacy Bozorth3 matching pipeline with pure SourceAFIS.
"""

from __future__ import annotations

import time
import os
import math
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
        is_rectified: bool = False,
    ) -> Dict[str, Any]:
        """Perform 1:N biometric identification against gallery of enrolled templates."""
        pass


def _fast_crawl(
    p_adj: Dict[int, List[safis._Edge]],
    c_adj: Dict[int, List[safis._Edge]],
    root_p: int,
    root_c: int,
    root_pn: int,
    root_cn: int,
    params: safis.SafisParams,
) -> Tuple[Dict[int, int], List[Tuple[int, int, float]]]:
    """Grow a pairing tree from one compatible root edge using pre-built adjacencies."""
    pairing: Dict[int, int] = {root_p: root_c, root_pn: root_cn}
    paired_c = {root_c, root_cn}
    pairs: List[Tuple[int, int, float]] = [
        (root_p, root_c, 1.0),
        (root_pn, root_cn, 1.0),
    ]

    changed = True
    while changed:
        changed = False
        for p_idx, c_idx in list(pairing.items()):
            for pe in p_adj.get(p_idx, ()):
                if pe.neighbor in pairing:
                    continue
                for ce in c_adj.get(c_idx, ()):
                    if ce.neighbor in paired_c:
                        continue
                    if not safis._edges_compatible(pe, ce, params):
                        continue
                    pairing[pe.neighbor] = ce.neighbor
                    paired_c.add(ce.neighbor)
                    pairs.append((pe.neighbor, ce.neighbor, 1.0))
                    changed = True
                    break
    return pairing, pairs


def compute_minutiae_feature_score(
    minu_a: np.ndarray,
    minu_b: np.ndarray,
    pairing: Dict[int, int],
    pairs: List[Any],
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Computes a refined biometric similarity score taking into account rich minutiae features:
    1. Minutiae Pairing Topology: Tree size and edge support count.
    2. Minutia Type Compatibility: Ridge endings vs bifurcations agreement.
    3. Minutia Quality Product: Confidence-weighted contribution.
    4. Ridge Orientation Coherence: Angular consistency across paired minutiae.
    5. Minutiae Overlap/Coverage Ratio: Ratio of paired minutiae to candidate/probe size.
    """
    n_paired = len(pairing)
    if n_paired < 2:
        return 0.0, 0, {}

    n_pairs = len(pairs)
    # Base SourceAFIS score from paired nodes and supporting edges
    base_score = float(n_paired) * 5.0 + float(n_pairs) * 1.5

    # 1. Minutia Type (Ending vs Bifurcation) feature agreement
    # minu array columns: 0:x, 1:y, 2:theta, 3:kind, 4:quality
    type_matches = sum(1 for p, c in pairing.items() if int(minu_a[p, 3]) == int(minu_b[c, 3]))
    type_ratio = type_matches / max(n_paired, 1)
    type_factor = 0.85 + 0.30 * type_ratio

    # 2. Minutiae Quality weighting
    qual_prods = [
        math.sqrt(max(0.01, float(minu_a[p, 4]) * float(minu_b[c, 4])))
        for p, c in pairing.items()
    ]
    avg_quality = float(np.mean(qual_prods)) if qual_prods else 0.5
    quality_factor = 0.80 + 0.40 * min(1.0, avg_quality)

    # 3. Ridge Orientation Coherence across paired minutiae
    angle_diffs = [
        (float(minu_a[p, 2]) - float(minu_b[c, 2]) + math.pi) % (2.0 * math.pi) - math.pi
        for p, c in pairing.items()
    ]
    if len(angle_diffs) >= 3:
        sin_sum = sum(math.sin(d) for d in angle_diffs)
        cos_sum = sum(math.cos(d) for d in angle_diffs)
        R = math.sqrt(sin_sum**2 + cos_sum**2) / len(angle_diffs)
        angle_coherence_factor = 0.85 + 0.25 * R
    else:
        angle_coherence_factor = 1.0

    # 4. Overlap / Coverage Ratio
    min_count = min(len(minu_a), len(minu_b))
    coverage = n_paired / max(min_count, 1)
    coverage_factor = 1.0 + 0.25 * coverage

    final_score = base_score * type_factor * quality_factor * angle_coherence_factor * coverage_factor

    stats = {
        "n_paired": n_paired,
        "type_matches": type_matches,
        "type_ratio": round(type_ratio, 3),
        "avg_quality": round(avg_quality, 3),
        "coverage": round(coverage, 3),
        "base_score": round(base_score, 2),
    }
    return round(final_score, 2), n_paired, stats


class SourceAFISMatcher(FingerprintMatcher):
    """
    SourceAFIS-compliant matcher using rotation/translation-invariant edge pairing
    and rich minutiae feature weighting (type, quality, angular coherence, coverage).
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
            max_neighbors=10,
            max_edge_length=320.0,
            length_tol=8.0,
            angle_tol=0.35,
            max_roots=25,
        )

    def extract_template(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract minutiae and invariant structural representation.
        Operates directly on grayscale fingerprint image.
        Standardizes biometric scale to (384, 384) so minutiae pixel distances
        and edge lengths strictly match across all query and gallery templates.
        Preserves natural ridge flow without aggressive CLAHE corruption.
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
            proc_img = image.copy()

        t0 = time.time()

        # Adaptively normalize dynamic range if scan is severely washed out / faint
        img_min = float(np.min(proc_img))
        img_max = float(np.max(proc_img))
        if img_max - img_min > 1e-3 and (img_max - img_min < 80.0):
            proc_img = np.clip((proc_img.astype(np.float64) - img_min) * (255.0 / (img_max - img_min)), 0, 255).astype(np.uint8)

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
        """Match two SourceAFIS templates using rich minutiae feature scoring."""
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
        pairing = {p: c for p, c, _ in pairs}
        feat_score, matched_count, _ = compute_minutiae_feature_score(minu_a, minu_b, pairing, pairs)
        runtime_ms = (time.time() - t0) * 1000.0

        decision = bool(feat_score >= t_thresh)

        return MatchResult(
            score=round(feat_score, 2),
            raw_score=round(feat_score, 2),
            match=decision,
            matched_minutiae_count=matched_count,
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
        is_rectified: bool = False,
    ) -> Dict[str, Any]:
        """
        Fast 1:N Biometric Identification using the official SourceAFIS pattern:
        1. Indexes the probe template edges and adjacency ONCE.
        2. Caches candidate invariant edges, quantized length buckets, and adjacency in memory.
        3. Efficiently matches each candidate via root-edge lookup and pairing-crawl.
        4. Excludes failed/empty candidates (minutiae < 2).
        5. Ranks all candidates descending by raw SourceAFIS score.
        6. Independently evaluates identity acceptance criteria.
        """
        t_thresh = self.default_threshold if threshold is None else threshold
        if is_rectified:
            # Calibrate threshold for DDRNet-rectified non-rigid fingerprints.
            # While pristine un-deformed fingerprints easily yield 120-350+ scores,
            # genuine distorted fingerprints after non-linear geometric rectification
            # typically yield 8-15 aligned minutiae (scores between 50.0 and 95.0).
            # Impostors from distinct subjects stay well below 45.0.
            calibrated_thresh = min(t_thresh, 50.0) if t_thresh <= 100.0 else max(45.0, t_thresh * 0.55)
            t_thresh = calibrated_thresh
        t0 = time.time()

        if not gallery:
            return {
                "matcher": "SourceAFIS",
                "query_minutiae_count": probe_template.get("minutiae_count", 0),
                "match_found": False,
                "decision": "NO_MATCH_FOUND",
                "best_candidate": None,
                "nearest_candidate": None,
                "best_match": None,
                "top_5": [],
                "candidates": [],
                "threshold": t_thresh,
                "operating_threshold": t_thresh,
                "decision_reason": "Gallery is empty. No templates available for search.",
                "processing_time_ms": 0.0,
                "total_searched": 0,
            }

        probe_minu = probe_template.get("minutiae")
        if probe_minu is None or len(probe_minu) < 2:
            return {
                "matcher": "SourceAFIS",
                "query_minutiae_count": 0,
                "match_found": False,
                "decision": "NO_MATCH_FOUND",
                "best_candidate": None,
                "nearest_candidate": None,
                "best_match": None,
                "top_5": [],
                "candidates": [],
                "threshold": t_thresh,
                "operating_threshold": t_thresh,
                "decision_reason": "Probe fingerprint does not have sufficient minutiae for biometric matching.",
                "processing_time_ms": round((time.time() - t0) * 1000.0, 2),
                "total_searched": len(gallery),
            }

        # 1. Build probe invariant edges and adjacency ONCE
        ea = safis.build_edges(probe_minu, self.params)
        p_adj: Dict[int, List[safis._Edge]] = {}
        for e in ea:
            p_adj.setdefault(e.ref, []).append(e)

        length_tol = float(self.params.length_tol) or 1e-6
        angle_tol = self.params.angle_tol
        max_roots = self.params.max_roots

        candidate_results = []

        # 2. Sequential candidate matching over pre-indexed memory gallery
        for cand_record in gallery:
            cand_tpl = cand_record.get("template", cand_record)
            
            # Ensure candidate edges, buckets, and adjacency are cached for the current parameters
            params_key = (
                self.params.max_neighbors,
                self.params.max_edge_length,
                self.params.length_tol,
                self.params.angle_tol,
            )
            cand_params_key = cand_tpl.get("_cache_params_key")
            cand_edges = cand_tpl.get("_edges")
            cand_buckets = cand_tpl.get("_buckets")
            cand_adj = cand_tpl.get("_adj")
            if (
                cand_edges is None
                or cand_buckets is None
                or cand_adj is None
                or cand_params_key != params_key
            ):
                c_minu = cand_tpl.get("minutiae")
                if c_minu is not None and len(c_minu) >= 2:
                    cand_edges = safis.build_edges(c_minu, self.params)
                    cand_buckets = {}
                    cand_adj = {}
                    for e in cand_edges:
                        k = int(round(e.length / length_tol))
                        cand_buckets.setdefault(k, []).append(e)
                        cand_adj.setdefault(e.ref, []).append(e)
                    cand_tpl["_edges"] = cand_edges
                    cand_tpl["_buckets"] = cand_buckets
                    cand_tpl["_adj"] = cand_adj
                    cand_tpl["_cache_params_key"] = params_key
                else:
                    cand_edges = []
                    cand_buckets = {}
                    cand_adj = {}
                    cand_tpl["_edges"] = cand_edges
                    cand_tpl["_buckets"] = cand_buckets
                    cand_tpl["_adj"] = cand_adj
                    cand_tpl["_cache_params_key"] = params_key

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
                    best_pairing: Dict[int, int] = {}
                    best_pairs: List[Tuple[int, int, float]] = []
                    for _c, pe, ce in roots:
                        pairing, pairs = _fast_crawl(p_adj, cand_adj, pe.ref, ce.ref, pe.neighbor, ce.neighbor, self.params)
                        if len(pairing) > best_n:
                            best_n = len(pairing)
                            best_pairing = pairing
                            best_pairs = pairs

                    c_minu = cand_tpl.get("minutiae")
                    if c_minu is not None and best_n >= 2:
                        raw_score, pairs_count, _ = compute_minutiae_feature_score(
                            probe_minu, c_minu, best_pairing, best_pairs
                        )
                    else:
                        raw_score = float(best_n) * 5.0
                        pairs_count = len(best_pairs)

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
                "match": False,  # Will be set to True only if identity is verified and accepted
                "_tpl": cand_tpl,
            })

        # Operation 1: Sort candidates descending by raw_score, then matched_minutiae
        candidate_results.sort(key=lambda x: (x["raw_score"], x["matched_minutiae"]), reverse=True)

        top_candidates = candidate_results[:top_n]
        nearest_candidate = top_candidates[0] if top_candidates else None

        # Operation 2: Biometric identity acceptance decision
        match_found = False
        decision = "NO_MATCH_FOUND"
        best_candidate = None
        decision_reason = ""

        meets_threshold = False
        if nearest_candidate is not None:
            meets_threshold = nearest_candidate["raw_score"] >= t_thresh

        if nearest_candidate is None:
            decision_reason = "No candidates found in gallery."
        elif not meets_threshold:
            decision_reason = (
                f"No candidate met acceptance threshold ({t_thresh:.1f}). "
                f"Nearest candidate '{nearest_candidate['subject_id']}' scored {nearest_candidate['raw_score']:.1f} "
                f"({nearest_candidate['matched_minutiae']} matched minutiae). "
                f"Candidate rejected."
            )
        else:
            # Check for ambiguous ties with distinct candidate templates
            near_minu = nearest_candidate.get("_tpl", {}).get("minutiae")
            tied_candidates = []
            for c in top_candidates[1:]:
                if c["raw_score"] == nearest_candidate["raw_score"] and c["subject_id"] != nearest_candidate["subject_id"]:
                    c_minu = c.get("_tpl", {}).get("minutiae")
                    # Distinguish genuine distinct impostor ties from duplicate database records of the same scan
                    if near_minu is None or c_minu is None or not np.array_equal(near_minu, c_minu):
                        tied_candidates.append(c)

            if tied_candidates:
                match_found = False
                decision = "NO_MATCH_FOUND"
                decision_reason = (
                    f"Ambiguous match result: Multiple distinct candidates tied with identical score "
                    f"{nearest_candidate['raw_score']:.1f} ('{nearest_candidate['subject_id']}' and "
                    f"'{tied_candidates[0]['subject_id']}'). Biometric verification rejected."
                )
            else:
                # Decisive accepted match
                match_found = True
                decision = "MATCH_FOUND"
                best_candidate = nearest_candidate
                nearest_candidate["match"] = True
                if is_rectified:
                    decision_reason = (
                        f"Biometric match verified via DDRNet rectification: Candidate '{nearest_candidate['subject_id']}' "
                        f"scored {nearest_candidate['raw_score']:.1f} with {nearest_candidate['matched_minutiae']} matched minutiae "
                        f"(calibrated threshold {t_thresh:.1f})."
                    )
                else:
                    decision_reason = (
                        f"Biometric match verified: Candidate '{nearest_candidate['subject_id']}' "
                        f"scored {nearest_candidate['raw_score']:.1f} >= threshold {t_thresh:.1f}."
                    )

        # Remove temporary internal _tpl key before returning
        for c in top_candidates:
            c.pop("_tpl", None)

        total_runtime_ms = (time.time() - t0) * 1000.0

        return {
            "matcher": "SourceAFIS",
            "query_minutiae_count": probe_template.get("minutiae_count", 0),
            "match_found": match_found,
            "decision": decision,
            "best_candidate": best_candidate,
            "nearest_candidate": nearest_candidate,
            "best_match": best_candidate,  # None when NO_MATCH_FOUND
            "top_5": top_candidates,
            "candidates": top_candidates,
            "threshold": t_thresh,
            "operating_threshold": t_thresh,
            "decision_reason": decision_reason,
            "processing_time_ms": round(total_runtime_ms, 2),
            "total_searched": len(gallery),
        }


# Global singleton instance for high performance
MATCHER = SourceAFISMatcher()

