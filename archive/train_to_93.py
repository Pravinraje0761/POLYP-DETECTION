"""
Alias for segmentation fine-tuning: boost Dice from ~90.86% to 93%+.
Trains on combined_dataset/ (images + masks).
Runs train_ultimate_93_plus_improved.py (loads best_adenseunet_boosted_fixed.pth).
"""
from train_ultimate_93_plus_improved import main

if __name__ == "__main__":
    main()
