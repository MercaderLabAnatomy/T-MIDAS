"""
Environment management utilities for T-MIDAS.
"""

import subprocess
import sys
from typing import Optional


def run_command(command: str, check: bool = True) -> str:
    """Run a shell command.
    
    Args:
        command: Command to execute
        check: Whether to raise exception on non-zero exit code
        
    Returns:
        Command output
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout


def check_environment(env_name: str) -> bool:
    """Check if a conda environment exists.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        True if environment exists
    """
    result = run_command("mamba env list", check=False)
    return env_name in result


def create_environment(env_name: str, python_version: str = "3.9") -> None:
    """Create a conda environment.
    
    Args:
        env_name: Name of the environment
        python_version: Python version to install
    """
    run_command(f"mamba create --name {env_name} python={python_version} -y")


def install_packages(env_name: str, packages: list[str]) -> None:
    """Install packages in environment.
    
    Args:
        env_name: Name of the environment
        packages: List of package names
    """
    for package in packages:
        run_command(f"mamba run -n {env_name} pip install {package}")


def setup_trackastra_env() -> None:
    """Set up TrackAstra environment."""
    env_name = "trackastra"
    if not check_environment(env_name):
        print("TrackAstra environment not found. Creating it now...")
        
        # Create commands to set up the environment
        commands = [
            f"mamba create --name {env_name} python=3.10 -y",
            f"mamba install -n {env_name} -c conda-forge -c gurobi -c funkelab ilpy -y",
            f"mamba install -n {env_name} -c conda-forge scikit-image numpy matplotlib tqdm tifffile -y",
            f"mamba run -n {env_name} pip install trackastra[ilp]"
        ]
        
        # Execute each command
        for cmd in commands:
            print(f"Executing: {cmd}")
            run_command(cmd)
        print("TrackAstra environment created successfully.")
    else:
        print("TrackAstra environment found.")
