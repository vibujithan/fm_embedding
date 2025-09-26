#!/usr/bin/env python3
import numpy as np
import os
import sys

def check_npy_files():
    files = [f for f in os.listdir('.') if f.endswith('.npy')]
    print(f'Total npy files: {len(files)}')
    
    errors = []
    issues = []
    
    for i, f in enumerate(files):
        try:
            data = np.load(f)
            print(f'{i+1:3d}/{len(files)}: {f}: OK - Shape: {data.shape}, dtype: {data.dtype}')
            
            # Check for potential issues
            if np.isnan(data).any():
                issues.append(f"{f}: Contains NaN values")
            if np.isinf(data).any():
                issues.append(f"{f}: Contains infinite values")
            if data.size == 0:
                issues.append(f"{f}: Empty array")
            if data.min() < 0:
                issues.append(f"{f}: Contains negative values (min: {data.min()})")
                
        except Exception as e:
            errors.append(f)
            print(f'{i+1:3d}/{len(files)}: {f}: ERROR - {e}')
    
    print(f'\n=== SUMMARY ===')
    print(f'Total files checked: {len(files)}')
    print(f'Files with errors: {len(errors)}')
    print(f'Files with issues: {len(issues)}')
    
    if errors:
        print(f'\n=== FILES WITH ERRORS ===')
        for e in errors:
            print(f'  {e}')
    
    if issues:
        print(f'\n=== FILES WITH ISSUES ===')
        for i in issues:
            print(f'  {i}')
    
    return len(errors) == 0 and len(issues) == 0

if __name__ == "__main__":
    success = check_npy_files()
    sys.exit(0 if success else 1) 