import inspect
from afis.pipeline import SafisExtractor, MindtctExtractor, FingerprintTemplate
import cv2

print("SafisExtractor methods:", dir(SafisExtractor))
print("SafisExtractor init sig:", inspect.signature(SafisExtractor.__init__))
print("SafisExtractor extract sig:", inspect.signature(SafisExtractor.extract))

# Test on 1.png
img_path = "data/clean_fvc/1.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
print("Image shape:", img.shape)

try:
    safis_ext = SafisExtractor()
    tpl_safis = safis_ext.extract(img)
    print("SafisExtractor output type:", type(tpl_safis))
    print("Isinstance FingerprintTemplate:", isinstance(tpl_safis, FingerprintTemplate))
    print("Minutiae shape:", tpl_safis.minutiae.shape)
    print("n_minutiae:", tpl_safis.n_minutiae if hasattr(tpl_safis, "n_minutiae") else len(tpl_safis.minutiae))
except Exception as e:
    print("SafisExtractor error:", e)

# Also check matcher.py
import sys
sys.path.append(".")
try:
    from matcher import SourceAFISMatcher
    matcher = SourceAFISMatcher()
    tpl_dict = matcher.extract_template(img)
    print("SourceAFISMatcher output keys:", tpl_dict.keys())
    print("SourceAFISMatcher minutiae count:", tpl_dict["minutiae_count"])
except Exception as e:
    print("SourceAFISMatcher error:", e)
