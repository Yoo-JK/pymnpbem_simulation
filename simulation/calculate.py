"""
Simulation Manager Class

This class orchestrates the MATLAB code generation process.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from .sim_utils.geometry_generator import GeometryGenerator
from .sim_utils.material_manager import MaterialManager
from .sim_utils.matlab_code_generator import MatlabCodeGenerator


class SimulationManager:
    """Manages the entire simulation generation process."""
    
    def __init__(self, config, verbose=False):
        """
        Initialize simulation manager.
        
        Args:
            config (dict): Configuration dictionary
            verbose (bool): Enable verbose output
        """
        self.config = config
        self.verbose = verbose
        self.matlab_code = None
        self.run_folder = None  # Will store the unique run folder path
        
        # Initialize sub-managers
        self.geometry_gen = GeometryGenerator(config, verbose)
        self.material_mgr = MaterialManager(config, verbose)
        self.matlab_gen = MatlabCodeGenerator(config, verbose)
        
        if verbose:
            print("SimulationManager initialized")
    
    def create_run_folder(self):
        """
        Create a unique folder for this simulation run.
        
        Returns:
            Path: Path to the created run folder
        """
        # Get base output directory
        base_output_dir = Path(self.config['output_dir'])
        
        # Create folder name (without timestamp)
        sim_name = self.config.get('simulation_name', 'simulation')
        folder_name = sim_name  # ← 타임스탬프 제거!
        
        self.run_folder = base_output_dir / folder_name
        
        # If folder already exists, remove it and create new one
        if self.run_folder.exists():
            if self.verbose:
                print(f"\n⚠ Folder already exists, will be overwritten: {self.run_folder}")
            shutil.rmtree(self.run_folder)
        
        # Create the folder
        self.run_folder.mkdir(parents=True, exist_ok=True)
        
        # Create logs subfolder
        (self.run_folder / 'logs').mkdir(exist_ok=True)
        
        if self.verbose:
            print(f"\n✓ Created run folder: {self.run_folder}")
        
        # Update config to use this run folder
        self.config['output_dir'] = str(self.run_folder)
        
        return self.run_folder
    
    def save_config_snapshot(self):
        """Save a snapshot of the configuration used for this run."""
        if self.run_folder is None:
            raise RuntimeError("Run folder not created. Call create_run_folder() first.")
        
        config_file = self.run_folder / 'config_snapshot.py'
        
        with open(config_file, 'w') as f:
            f.write("# Configuration snapshot\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("args = {\n")
            for key, value in self.config.items():
                if isinstance(value, str):
                    f.write(f"    '{key}': '{value}',\n")
                else:
                    f.write(f"    '{key}': {value},\n")
            f.write("}\n")
        
        if self.verbose:
            print(f"✓ Saved config snapshot: {config_file}")
        
        return str(config_file)
    
    def generate_matlab_code(self):
        """Generate complete MATLAB simulation code."""
        if self.verbose:
            print("\n--- Generating MATLAB Code ---")
        
        # Generate geometry code
        if self.verbose:
            print("Generating geometry code...")
        geometry_code = self.geometry_gen.generate()
        
        # Generate material code
        if self.verbose:
            print("Generating material code...")
        material_code = self.material_mgr.generate()
        
        # Generate complete MATLAB script
        if self.verbose:
            print("Assembling complete MATLAB script...")
        self.matlab_code = self.matlab_gen.generate_complete_script(
            geometry_code=geometry_code,
            material_code=material_code
        )
        
        if self.verbose:
            print("MATLAB code generation complete")
        
        return self.matlab_code
    
    def save_matlab_script(self, output_path=None):
        """
        Save MATLAB script to file.
        
        Args:
            output_path (str): Output file path. If None, saves to run folder if available,
                             otherwise uses default location.
        
        Returns:
            str: Path to saved file
        """
        if self.matlab_code is None:
            raise RuntimeError("MATLAB code not generated yet. Call generate_matlab_code() first.")
        
        # Determine output path
        if output_path is None:
            if self.run_folder is not None:
                # Save to run folder
                output_path = self.run_folder / 'simulation_script.m'
            else:
                # Fall back to default location
                output_dir = Path(__file__).parent
                output_path = output_dir / 'simulation_script.m'
        else:
            output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        with open(output_path, 'w') as f:
            f.write(self.matlab_code)
        
        if self.verbose:
            print(f"✓ MATLAB script saved to: {output_path}")
        
        return str(output_path)
    
    def get_run_folder(self):
        """Get the run folder path."""
        return self.run_folder
    
    def get_summary(self):
        """Get summary of simulation configuration."""
        summary = {
            'structure': self.config['structure'],
            'simulation_type': self.config['simulation_type'],
            'excitation': self.config['excitation_type'],
            'wavelength_range': self.config['wavelength_range'],
            'materials': self.config['materials'],
            'run_folder': str(self.run_folder) if self.run_folder else None
        }
        return summary