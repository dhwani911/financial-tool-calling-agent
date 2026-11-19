import json
from types import SimpleNamespace

from app.agent import get_function_calls


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