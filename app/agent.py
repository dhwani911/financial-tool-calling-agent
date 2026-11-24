from openai import OpenAIError

from app.openai_client import (
    create_openai_client,
    get_openai_model,
)
from app.tool_definitions import get_openai_tools
import json
from pydantic import ValidationError
from app.schemas import ROIInput, ProfitInput, CompoundInterestInput

SYSTEM_INSTRUCTIONS = """
You are a financial calculation assistant.

You have access to financial calculation tools.

Use the available tools when the user requests a financial
calculation.

Do not invent calculation results.

The application will execute the requested tools separately.
"""

TOOL_SCHEMAS = {
    "calculate_roi": ROIInput,
    "calculate_profit": ProfitInput,
    "calculate_compound_interest": CompoundInterestInput,
}

def send_financial_question(question: str):
    """
    Send a financial question and available tools to OpenAI.

    This function only requests and inspects the model response.
    It does not execute the requested Python financial function.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Financial question must be a non-empty string."
        )

    # Initialize the OpenAI client
    client = create_openai_client()

    # Retrieve the model name.
    model = get_openai_model()        # All the financial calculation tools available for the model to use.

    # Load available Finance tools.
    tools = get_openai_tools()          # It will return all the available tools for financial calculations, including ROI, Profit, and Compound Interest.

    # Send the question and tool definitions to OpenAI
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=question.strip(),
        tools=tools,
        tool_choice="required",
    )

    return response


def get_function_calls(response) -> list:
    """
    Detect function calls in an OpenAI Responses API response.

    Returns a list of function-call output items.

    An empty list means no function call was requested.
    """

    output_items = getattr(response, "output", None) or []

    function_calls = [
        item
        for item in output_items
        if getattr(item, "type", None) == "function_call"
    ]

    return function_calls


def display_response(response) -> None:
    """
    Display the OpenAI response and detect requested function calls.
    """

    print("\nResponse ID:", response.id)
    print("Response status:", response.status)
    
    try:
        tool_calls = extract_tool_calls(response)
    
        if tool_calls:    
            print("\nOpenAI requested the following tool(s):")
    
            for tool_call in tool_calls:            
                print("\nTool name:", tool_call["tool_name"])
                print("Call ID:", tool_call["call_id"])
                print("Raw arguments:", json.dumps(tool_call["arguments"], indent=4))         # print the arguments in a pretty JSON format for better readability
    
        else:    
            output_text = getattr(response, "output_text", None)    
            if output_text:
                print("\nOpenAI response:")
                print(output_text)    
            else:
                print("\nNo function call or text response was returned.")
    except ValueError as exc:
        print(f"\nError processing function calls as Tool extraction failed: {exc}")

def extract_tool_call(call) -> dict:
    """
    Extract the tool name, arguments, and call ID
    from an OpenAI function-call object.

    Returns a dictionary containing the extracted data.
    """

    # Verify that this is a function call.
    if getattr(call, "type", None) != "function_call":
        raise ValueError("Invalid response: expected a function call.")

    # Extract the function name.
    tool_name = getattr(call, "name", None)

    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("Function call is missing a valid tool name.")

    # Extract the unique call ID.
    call_id = getattr(call, "call_id", None)

    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("Function call is missing a valid call ID.")

    # Extract and parse JSON arguments.
    raw_arguments = getattr(call, "arguments", None)

    if not isinstance(raw_arguments, str):
        raise ValueError("Function call arguments must be a JSON string.")

    try:
        arguments = json.loads(raw_arguments)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON arguments for tool '{tool_name}'."
        ) from exc

    # Tool arguments must be a JSON object.
    if not isinstance(arguments, dict):
        raise ValueError(
            f"Arguments for tool '{tool_name}' must be a JSON object."
        )

    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "call_id": call_id,
    }

def validate_tool_arguments(tool_call: dict) -> dict:
    """
    Validate OpenAI-generated function arguments using Pydantic.

    Args:
        tool_call: Extracted tool call containing tool_name,
                   arguments, and call_id.

    Returns:
        Dictionary containing validated arguments.

    Raises:
        ValueError: If the tool name is unsupported.
        ValidationError: If arguments fail Pydantic validation.
    """
    
    tool_name = tool_call.get("tool_name")
    arguments = tool_call.get("arguments")

    if tool_name not in TOOL_SCHEMAS:
        raise ValueError(f"Unsupported tool name: {tool_name}")

    schema = TOOL_SCHEMAS[tool_name]    
    validated_arguments = schema.model_validate(arguments)          # model_validate() is a Pydantic method that validates the Pydantic model instance.

    return {"tool_name": tool_name, "arguments": validated_arguments.model_dump(), "call_id": tool_call.get("call_id")}       # model_dump() generates a dictionary representation of the model.

def main() -> None:
    """Run a sample financial question."""

    question = input(
        "\nEnter your financial question: "
    ).strip()

    if not question:
        print("Please enter a valid question.")
        return

    try:
        response = send_financial_question(question)

        display_response(response)

    except OpenAIError as exc:
        print(f"\nOpenAI API request failed: {exc}")

    except RuntimeError as exc:
        print(f"\nConfiguration error: {exc}")

    except ValueError as exc:
        print(f"\nInput error: {exc}")

def extract_tool_calls(response) -> list[dict]:
    """
    Extract tool call information from a list of OpenAI function-call objects.

    Returns a list of dictionaries containing tool name, arguments, and call ID.
    """

    function_call = get_function_calls(response)
    
    return [
        extract_tool_call(call)
        for call in function_call
    ]
    

if __name__ == "__main__":
    main()