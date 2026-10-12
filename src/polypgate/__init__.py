"""polypgate: code for "When More Preprocessing Hurts: Distribution Shift and Failure of
Segmentation-Guided Colorectal Polyp Classification" (Pravin Raj E and Nishanthini).

Subpackages
-----------
models         A-DenseUNet segmenter and ViT classifier
losses         Focal Tversky + boundary segmentation loss
metrics        segmentation, classification and explainability metrics
data           dataset loaders
preprocessing  soft ROI, selective RGB enhancement, LAB-CLAHE
explain        Grad-CAM and attention rollout
"""

__version__ = "0.1.0"
