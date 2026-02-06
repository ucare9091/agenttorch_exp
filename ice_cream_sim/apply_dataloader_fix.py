#!/usr/bin/env python3
"""
Patch script to fix dataloader.py TypeError for numpy.object arrays.
This script applies the _fix_df function fix to agent_torch/core/dataloader.py
"""

import os
import sys
import site
from pathlib import Path

def find_dataloader_path():
    """Find the dataloader.py file in agent_torch package"""
    # Check common locations
    possible_paths = []
    
    # Check site-packages
    for site_dir in site.getsitepackages():
        dataloader_path = Path(site_dir) / "agent_torch" / "core" / "dataloader.py"
        if dataloader_path.exists():
            possible_paths.append(dataloader_path)
    
    # Check in current Python environment
    import site
    for site_dir in site.getsitepackages():
        dataloader_path = Path(site_dir) / "agent_torch" / "core" / "dataloader.py"
        if dataloader_path.exists() and dataloader_path not in possible_paths:
            possible_paths.append(dataloader_path)
    
    if not possible_paths:
        # Try to import and find the module
        try:
            import agent_torch.core.dataloader as dl_module
            module_path = Path(dl_module.__file__)
            if module_path.exists():
                possible_paths.append(module_path)
        except ImportError:
            pass
    
    if possible_paths:
        return possible_paths[0]
    
    return None

def apply_fix(dataloader_path):
    """Apply the fix to dataloader.py"""
    dataloader_path = Path(dataloader_path)
    
    if not dataloader_path.exists():
        print(f"Error: {dataloader_path} not found")
        return False
    
    # Read the file
    with open(dataloader_path, 'r') as f:
        content = f.read()
    
    # Check if fix already applied
    if '_fix_df' in content and 'torch.tensor(LoadPopulation._fix_df(df).values' in content:
        print(f"Fix already applied to {dataloader_path}")
        return True
    
    # Check if numpy import exists
    needs_numpy_import = 'import numpy as np' not in content
    
    # Add numpy import if needed
    if needs_numpy_import:
        content = content.replace(
            'import pandas as pd\nimport torch',
            'import pandas as pd\nimport numpy as np\nimport torch'
        )
    
    # Find the load_population method in LoadPopulation class
    # Look for the pattern: "def load_population(self):" followed by the problematic line
    old_pattern = """    def load_population(self):
        pickle_files = glob.glob(
            f"{self.population_folder_path}/*.pickle", recursive=False
        )
        for file in pickle_files:
            with open(file, "rb") as f:
                key = os.path.splitext(os.path.basename(file))[0]
                df = pd.read_pickle(file)
                setattr(self, key, torch.from_numpy(df.values).float())"""
    
    new_code = """    @staticmethod
    def _fix_df(df):
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().sum() > len(df) * 0.5:
                    df[col] = converted.fillna(0).astype(np.float32)
                else:
                    df[col] = pd.Categorical(df[col]).codes.astype(np.int32)
            elif df[col].dtype == 'bool':
                df[col] = df[col].astype(np.int8)
        return df

    def load_population(self):
        pickle_files = glob.glob(
            f"{self.population_folder_path}/*.pickle", recursive=False
        )
        for file in pickle_files:
            with open(file, "rb") as f:
                key = os.path.splitext(os.path.basename(file))[0]
                df = pd.read_pickle(file)
                tensor = torch.tensor(LoadPopulation._fix_df(df).values, dtype=torch.float32)
                setattr(self, key, tensor)"""
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_code)
        
        # Write back
        with open(dataloader_path, 'w') as f:
            f.write(content)
        
        print(f"Fix applied successfully to {dataloader_path}")
        return True
    else:
        print(f"Error: Could not find the expected pattern in {dataloader_path}")
        print("The file may have been modified or is in a different format.")
        return False

def main():
    print("Applying dataloader.py fix...")
    
    dataloader_path = find_dataloader_path()
    
    if not dataloader_path:
        print("Error: Could not find agent_torch/core/dataloader.py")
        print("Please ensure agent_torch is installed in your Python environment.")
        sys.exit(1)
    
    print(f"Found dataloader.py at: {dataloader_path}")
    
    if apply_fix(dataloader_path):
        print("Fix applied successfully!")
        sys.exit(0)
    else:
        print("Failed to apply fix.")
        sys.exit(1)

if __name__ == "__main__":
    main()
