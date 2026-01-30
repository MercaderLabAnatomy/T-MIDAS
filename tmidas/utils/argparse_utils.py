"""
Common argument parsing utilities for T-MIDAS scripts.
"""

import argparse
from typing import Any


def create_parser(description: str, input_help: str = "Path to input folder") -> argparse.ArgumentParser:
    """Create a basic argument parser.
    
    Args:
        description: Description for the parser
        input_help: Help text for the input argument
        
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=str, required=True, help=input_help)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common arguments to parser.
    
    Args:
        parser: Argument parser to modify
        
    Returns:
        Modified argument parser
    """
    parser.add_argument("--output", type=str, help="Path to output folder")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser
