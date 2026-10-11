import sys
sys.path.append(".")
import cv2
import pickle
import afis
import afis.pipeline as pipeline
from matcher import SourceAFISMatcher

print("afis version:", getattr(afis, "__version__", "unknown"))

img = cv2.imread("data/clean_fvc/1.png", cv2.IMREAD_GRAYSCALE)
matcher = SourceAFISMatcher()
res = matcher.extract_template(img)
print("matcher.extract_template minutiae count:", res["minutiae_count"])
print("matcher.extract_template version:", res["version"])
print("matcher.extract_template minutiae shape:", res["minutiae"].shape)

# Let's inspect SafisExtractor in afis.pipeline
try:
    se = pipeline.SafisExtractor()
    print("SafisExtractor instance dir:", [d for d in dir(se) if not d.startswith("__")])
    # Call extract_minutiae
    out = se.extract_minutiae(img)
    print("SafisExtractor.extract_minutiae type:", type(out))
    if hasattr(out, "minutiae"):
        print("SafisExtractor minutiae shape:", out.minutiae.shape)
        print("SafisExtractor class:", out.__class__.__name__)
except Exception as e:
    print("SafisExtractor error:", e)
