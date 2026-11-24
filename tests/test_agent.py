import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent import get_function_calls, extract_tool_calls, extract_tool_call, validate_tool_arguments, \
    validate_and_execute_tool


def test_detect_single_function_call():
    """Detect one function call in the response."""
	
    roi_call = SimpleNamespace(
        type="function_call",
        name="calculate_roi",
        call_id="call_123",
        arguments='{"investment": 10000, "profit": 2500}',
    )

    response = SimpleNamespace(
        output=[roi_call]
    )

    calls = get_function_calls(response)

    assert len(calls) == 1
    assert calls[0].name == "calculate_roi"
    assert calls[0].call_id == "call_123"
    assert json.loads(calls[0].arguments)["investment"] == 10000

def test_detect_multiple_function_calls():
    """Detect multiple function calls in one response."""

    roi_call = SimpleNamespace(
        type="function_call",
        name="calculate_roi",
        call_id="call_123",
        arguments='{"investment": 10000, "profit": 2500}',
    )

    profit_call = SimpleNamespace(
        type="function_call",
        name="calculate_profit",
        call_id="call_456",
        arguments='{"revenue": 20000, "cost": 12000}',
    )

    response = SimpleNamespace(
        output=[roi_call, profit_call]
    )
 
    calls = get_function_calls(response)

    assert len(calls) == 2

    assert calls[0].name == "calculate_roi"
    assert calls[1].name == "calculate_profit"
    assert json.loads(calls[0].arguments)["profit"] == 2500
    assert json.loads(calls[1].arguments)["cost"] == 12000


def test_no_function_calls_in_response():
    """An ordinary text response should return an empty list."""

    message = SimpleNamespace(
        type="message"
    )

    response = SimpleNamespace(
        output=[message]
    )

    calls = get_function_calls(response)

    assert calls == []


def test_response_with_mixed_output_items():
    """Only function-call items should be detected."""

    message = SimpleNamespace(
        type="message"
    )

    reasoning = SimpleNamespace(
        type="reasoning"
    )

    roi_call = SimpleNamespace(
        type="function_call",
        name="calculate_roi",
        call_id="call_789",
        arguments='{"investment": 10000, "profit": 2500}',
    )

    response = SimpleNamespace(
        output=[
            reasoning,
            message,
            roi_call,
        ]
    )

    calls = get_function_calls(response)

    assert len(calls) == 1
    assert calls[0].name == "calculate_roi"


def test_missing_output_returns_empty_list():
    """Missing output should not cause an exception."""

    response = SimpleNamespace()

    calls = get_function_calls(response)

    assert calls == []
    
def test_extract_tool_call_successfully():
    """Extract tool name, arguments, and call ID."""

    call = SimpleNamespace(
        type="function_call",
        name="calculate_roi",
        call_id="call_123",
        arguments='{"investment": 10000, "profit": 2500}',
    )

    result = extract_tool_call(call)

    assert result["tool_name"] == "calculate_roi"
    assert result["arguments"] == {
        "investment": 10000,
        "profit": 2500,
    }
    assert result["call_id"] == "call_123"
    
def test_extract_multiple_tool_calls_successfully():
    """Extract multiple tool calls from a response."""

    roi_call = SimpleNamespace(
        type="function_call",
        name="calculate_roi",
        call_id="call_123",
        arguments='{"investment": 10000, "profit": 2500}',
    )
    profit_call = SimpleNamespace(
        type="function_call",
        name="calculate_profit",
        call_id="call_456",
        arguments='{"revenue": 20000, "cost": 12000}',
    )
    response = SimpleNamespace(
        output=[roi_call, profit_call]
    )

    tool_calls = extract_tool_calls(response)

    assert len(tool_calls) == 2
    assert tool_calls[0]["tool_name"] == "calculate_roi"
    assert tool_calls[1]["tool_name"] == "calculate_profit"
    
def test_extract_tool_rejects_invalid_json():
    """Invalid JSON in arguments should raise a ValueError."""

    call = SimpleNamespace(
        type="function_call",
        name="calculate_roi",
        call_id="call_123",
        arguments='{"investment": 10000, "profit": }',  # Missing closing brace
    )

    with pytest.raises(ValueError, match="Invalid JSON arguments for tool 'calculate_roi'") as exc_info:
        extract_tool_call(call)
        
def test_extract_tool_call_rejects_non_object_arguments():
    """Arguments must be a JSON object, not an array or other type."""

    call = SimpleNamespace(
        type="function_call",
        name="calculate_roi",
        call_id="call_123",
        arguments='[10000, 2500]',  # JSON array instead of object
    )

    with pytest.raises(ValueError, match="JSON object") as exc_info:
        extract_tool_call(call)
        
def test_extract_tool_call_rejects_missing_name_or_call_id():
    """Missing tool name or call ID should raise a ValueError."""

    # Missing name
    call_missing_name = SimpleNamespace(
        type="function_call",
        call_id="call_123",
        arguments='{"investment": 10000, "profit": 2500}',
    )

    with pytest.raises(ValueError, match="missing a valid tool name") as exc_info:
        extract_tool_call(call_missing_name)

    # Missing call ID
    call_missing_id = SimpleNamespace(
        type="function_call",
        name="calculate_roi",
        arguments='{"investment": 10000, "profit": 2500}',
    )

    with pytest.raises(ValueError, match="missing a valid call ID") as exc_info:
        extract_tool_call(call_missing_id)
        
def test_extract_tool_call_rejects_invalid_type():
    """If the call type is not 'function_call', it should raise a ValueError."""

    call_invalid_type = SimpleNamespace(
        type="message",      # Not a function call
        name="calculate_roi",
        call_id="call_123",
        arguments='{"investment": 10000, "profit": 2500}',
    )

    with pytest.raises(ValueError, match="expected a function call") as exc_info:
        extract_tool_call(call_invalid_type)
        
def test_validate_valid_roi_arguments():
    """Valid ROI arguments should pass validation."""
    
    tool_call = {
        "tool_name": "calculate_roi",
        "arguments": {
            "investment": 10000,
            "profit": 2500,
        },
        "call_id": "call_123",
    }

    validated_args = validate_tool_arguments(tool_call)
    
    assert validated_args['tool_name'] == "calculate_roi"
    assert validated_args['arguments']["investment"] == 10000
    assert validated_args['arguments']["profit"] == 2500
    
def test_validate_valid_profit_arguments():
    """Valid Profit arguments should pass validation."""
    
    tool_call = {
        "tool_name": "calculate_profit",
        "arguments": {
            "revenue": 20000,
            "cost": 12000,
        },
        "call_id": "call_456",
    }

    validated_args = validate_tool_arguments(tool_call)
    
    assert validated_args['tool_name'] == "calculate_profit"
    assert validated_args['arguments']["revenue"] == 20000
    assert validated_args['arguments']["cost"] == 12000
    
def test_validate_valid_compound_interest_arguments():
    """Valid Compound Interest arguments should pass validation."""
    
    tool_call = {
        "tool_name": "calculate_compound_interest",
        "arguments": {
            "principal": 10000,
            "annual_rate": 5,
            "years": 3,
        },
        "call_id": "call_789",
    }

    validated_args = validate_tool_arguments(tool_call)
    
    assert validated_args['tool_name'] == "calculate_compound_interest"
    assert validated_args['arguments']["principal"] == 10000
    assert validated_args['arguments']["annual_rate"] == 5
    assert validated_args['arguments']["years"] == 3
    
def test_validate_invalid_roi_arguments():
    """Invalid ROI arguments should raise a ValidationError."""
    
    tool_call = {
        "tool_name": "calculate_roi",
        "arguments": {
            "investment": -10000,  # Invalid negative investment
            "profit": 2500,
        },
        "call_id": "call_123",
    }

    with pytest.raises(ValidationError):
        validate_tool_arguments(tool_call)
        
def test_validate_missing_required_argument():
    """Missing required argument should raise a ValidationError."""
    
    tool_call = {
        "tool_name": "calculate_profit",
        "arguments": {
            "revenue": 20000,
            # Missing 'cost'
        },
        "call_id": "call_456",
    }

    with pytest.raises(ValidationError):
        validate_tool_arguments(tool_call)
        
def test_validate_unexpected_argument():
    """Unexpected argument should raise a ValidationError."""
    
    tool_call = {
        "tool_name": "calculate_compound_interest",
        "arguments": {
            "principal": 10000,
            "annual_rate": 5,
            "years": 3,
            "unexpected_param": 42,  # Extra unexpected parameter
        },
        "call_id": "call_789",
    }

    with pytest.raises(ValidationError):
        validate_tool_arguments(tool_call)
        
def test_validate_unsupported_tool():
    """Unsupported tool name should raise a ValueError."""
    
    tool_call = {
        "tool_name": "unsupported_tool",
        "arguments": {
            "some_param": 123,
        },
        "call_id": "call_000",
    }

    with pytest.raises(ValueError, match="Unsupported tool name"):
        validate_tool_arguments(tool_call)
        
def test_execute_roi_tool():
    """Execute the ROI tool with valid arguments."""
    
    tool_call = {
        "tool_name": "calculate_roi",
        "arguments": {
            "investment": 10000,
            "profit": 2500,
        },
        "call_id": "call_123",
    }

    result = validate_and_execute_tool(tool_call)    

    assert result["tool_name"] == "calculate_roi"
    assert result["result"] == 25.0
    assert result["call_id"] == "call_123"
    
def test_execute_profit_tool():
    """Execute the Profit tool with valid arguments."""
    
    tool_call = {
        "tool_name": "calculate_profit",
        "arguments": {
            "revenue": 20000,
            "cost": 12000,
        },
        "call_id": "call_456",
    }

    result = validate_and_execute_tool(tool_call)    

    assert result["tool_name"] == "calculate_profit"
    assert result["result"] == 8000
    assert result["call_id"] == "call_456"
    
def test_execute_compound_interest_tool():
    """Execute the Compound Interest tool with valid arguments."""
    
    tool_call = {
        "tool_name": "calculate_compound_interest",
        "arguments": {
            "principal": 10000,
            "annual_rate": 5,
            "years": 3,
        },
        "call_id": "call_789",
    }

    result = validate_and_execute_tool(tool_call)    

    assert result["tool_name"] == "calculate_compound_interest"
    assert round(result["result"], 2) == 11576.25
    assert result["call_id"] == "call_789"
    
def test_execute_unsupported_tool():
    """Executing an unsupported tool should raise a ValueError."""
    
    tool_call = {
        "tool_name": "unsupported_tool",
        "arguments": {
            "some_param": 123,
        },
        "call_id": "call_000",
    }

    with pytest.raises(ValueError, match="Unsupported tool name"):
        validate_and_execute_tool(tool_call)
        
def test_execute_invalid_roi_arguments():
    """Executing ROI tool with invalid arguments should raise a ValidationError."""
    
    tool_call = {
        "tool_name": "calculate_roi",
        "arguments": {
            "investment": -10000,  # Invalid negative investment
            "profit": 2500,
        },
        "call_id": "call_123",
    }

    with pytest.raises(ValidationError):
        validate_and_execute_tool(tool_call)
        
def test_execute_unexpected_arguments():
    """Executing a tool with unexpected arguments should raise a ValidationError."""
    
    tool_call = {
        "tool_name": "calculate_profit",
        "arguments": {
            "revenue": 20000,
            "cost": 12000,
            "unexpected_param": 42,  # Extra unexpected parameter
        },
        "call_id": "call_456",
    }

    with pytest.raises(ValidationError):
        validate_and_execute_tool(tool_call)