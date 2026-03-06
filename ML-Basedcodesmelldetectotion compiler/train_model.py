import os
import random
from ml_code_smell.features import FeatureExtractor
from ml_code_smell.model import SmellDetectorModel
import numpy as np

def generate_synthetic_data(num_samples=100):
    """
    Generates synthetic Python code snippets and labels them.
    Label 1: Smelly (e.g., long functions, deep nesting)
    Label 0: Clean
    """
    snippets = []
    labels = []

    for _ in range(num_samples // 2):
        clean_code = "def clean_function():\n"
        clean_code += "    x = 1\n"
        clean_code += "    y = 2\n"
        clean_code += "    return x + y\n"
        snippets.append(clean_code)
        labels.append(0)

    
        smelly_code = "def complex_function():\n"
        for i in range(40):
            smelly_code += f"    var_{i} = {i}\n"
        
        smelly_code += "    if True:\n"
        smelly_code += "        if True:\n"
        smelly_code += "            if True:\n"
        smelly_code += "                if True:\n"
        smelly_code += "                    print('Deep')\n"
        
        snippets.append(smelly_code)
        labels.append(1)

    return snippets, labels

def main():
    print("Generating synthetic dataset...")
    snippets, labels = generate_synthetic_data(200)
    
    print("Extracting features...")
    extractor = FeatureExtractor()
    X = []
    for code in snippets:
        features = extractor.extract_features(code)
        X.append(features)
    
    print("Training model...")
    model = SmellDetectorModel()
    model.train(X, labels)
    
    print("Saving model...")
    model.save()
    print("Done!")

if __name__ == "__main__":
    main()
