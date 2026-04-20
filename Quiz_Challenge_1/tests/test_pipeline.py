import os
import pandas as pd
import pytest
import sys

# Add project root to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generator import PromptEngine

def test_dataset_loading():
    """Ensure products.csv exists and has exactly 5 products."""
    data_path = os.path.join("data", "products.csv")
    assert os.path.exists(data_path), "products.csv not found"
    
    df = pd.read_csv(data_path)
    assert len(df) == 5, f"Expected 5 products, found {len(df)}"
    assert list(df.columns) == ["product_id", "title", "category", "color", "material", "style"]

def test_prompt_engine_baseline():
    """Ensure baseline prompt uses only the title."""
    engine = PromptEngine()
    row = {"title": "Test Item"}
    assert engine.generate_baseline_prompt(row) == "Test Item"

def test_prompt_engine_improved():
    """Ensure improved prompt contains all metadata tags."""
    engine = PromptEngine()
    row = {
        "title": "Sun Hat",
        "color": "yellow",
        "material": "straw",
        "style": "beach"
    }
    prompt = engine.generate_improved_prompt(row)
    assert "yellow" in prompt
    assert "straw" in prompt
    assert "Sun Hat" in prompt
    assert "beach" in prompt
    assert "Professional studio photography" in prompt
    assert "pure white solid background" in prompt

def test_metadata_consistency():
    """Verify that products with identical metadata (except title) generate consistent improved prompts."""
    engine = PromptEngine()
    
    # Product 1 and 4 are both red cotton casual
    p1 = {"title": "Dress", "color": "red", "material": "cotton", "style": "casual"}
    p4 = {"title": "T-Shirt", "color": "red", "material": "cotton", "style": "casual"}
    
    prompt1 = engine.generate_improved_prompt(p1)
    prompt4 = engine.generate_improved_prompt(p4)
    
    # Check if the core attributes are identical
    # They should only differ by the title string
    assert "red cotton Dress, casual" in prompt1
    assert "red cotton T-Shirt, casual" in prompt4
    
    # If we replace titles, the prompts should be identical
    assert prompt1.replace("Dress", "TEMP") == prompt4.replace("T-Shirt", "TEMP")

def test_negative_prompt():
    """Ensure negative prompt is non-empty and contains quality tags."""
    engine = PromptEngine()
    neg = engine.get_negative_prompt()
    assert len(neg) > 0
    assert "blurry" in neg
    assert "low quality" in neg
