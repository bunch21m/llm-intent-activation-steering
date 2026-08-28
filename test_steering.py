import torch
from steering import *
from langchain_core.messages import HumanMessage

def test_model_loading():
    print("Testing model loading...")
    assert isinstance(response_model, SteeredModelWrapper)
    print("Model wrapper initialized successfully.")
    print(f"Device: {response_model.device}")
    
def test_generation():
    print("Testing generation (clean)...")
    msg = HumanMessage(content="Hello, how are you?")
    response = response_model.invoke([msg])
    print(f"Response: {response.content}")
    assert len(response.content) > 0

def test_cat():
    print("Testing CAT steering...")
    # Register the hook by calling the tool
    print(mention_cat_tool.invoke({}))
    
    assert "cat" in response_model.active_steering
    
    # Generate with steer
    msg = HumanMessage(content="This is a test. What store should I go to?")
    response = response_model.invoke([msg])
    print(f"Steered Response: {response.content}")
    
    # Reset
    response_model.clear_hooks()
    assert len(response_model.hooks) == 0


def test_joyful():
    print("Testing joyful steering...")
    # Register the hook by calling the tool
    print(joyful_tool.invoke({}))
    
    assert "joyful" in response_model.active_steering
    
    # Generate with steer
    msg = HumanMessage(content="This is a test. What store should I go to?")
    response = response_model.invoke([msg])
    print(f"Steered Response: {response.content}")
    
    # Reset
    response_model.clear_hooks()
    assert len(response_model.hooks) == 0

def test_lower_confidence():
    print("Testing lower confidence steering...")
    # Register the hook by calling the tool
    print(lower_confidence_tool.invoke({}))
    
    assert "uncertainty" in response_model.active_steering
    
    # Generate with steer
    msg = HumanMessage(content="This is a test. What store should I go to?")
    response = response_model.invoke([msg])
    print(f"Steered Response: {response.content}")
    
    # Reset
    response_model.clear_hooks()
    assert len(response_model.hooks) == 0

if __name__ == "__main__":
    test_model_loading()
    test_generation()
    test_cat()
    test_joyful()
    test_lower_confidence()
