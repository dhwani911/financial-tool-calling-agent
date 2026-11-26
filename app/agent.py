import re

from openai import OpenAIError

from app.openai_client import (
    create_openai_client,
    get_openai_model,
)
from app.tool_definitions import get_openai_tools
import json
from pydantic import ValidationError
from app.schemas import ROIInput, ProfitInput, CompoundInterestInput
from app.tools import calculate_roi, calculate_profit, calculate_compound_interest

SYSTEM_INSTRUCTIONS = """
You are a helpful financial calculation assistant.

Your responsibilities:
1. Understand the user's financial questions.
2. Use the available financial tools for calculations.
3. Never invent or estimate calculation results when a tool is available.
4. Explain calculation results clearly and accurately.
5. Present percentages with the % symbol.
6. Present monetary values with appropriate currency formatting
   when the currency is known.
7. Explain the result in simple, user-friendly language.
8. Do not claim to provide professional financial advice.
9. If a required input is missing, ask the user for clarification.

Always base numerical answers on the actual tool results.
"""

TOOL_SCHEMAS = {
    "calculate_roi": ROIInput,
    "calculate_profit": ProfitInput,
    "calculate_compound_interest": CompoundInterestInput,
}

TOOL_REGISTRY = {
    "calculate_roi": calculate_roi,
    "calculate_profit": calculate_profit,
    "calculate_compound_interest": calculate_compound_interest,
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
        tool_choice="auto",                   
    )

    return response


def get_function_calls(response) -> list:
    """
    Detect function calls in an OpenAI Responses API response.

    Returns a list of function-call output items.

    An empty list means no function call was requested.
    """

    output_items = getattr(response, "output", None) or []          
    
    # The function/tool or called has type = "function_call", rest other tools carries type="function" in response of OpenAI api
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

def execute_tool(tool_call: dict) -> dict:
    """
    Execute the validated Financial tool.

    Args:
        tool_call: Validated tool call dict containing tool_name, validated arguments, and call_id.

    Returns:
        Dictionary containing the tool name, result, and call ID.

    Raises:
        ValueError: If the tool name is unsupported or execution fails.
    """

    tool_name = tool_call.get("tool_name")
    arguments = tool_call.get("arguments")
    call_id = tool_call.get("call_id")

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unsupported tool name: {tool_name}")

    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for tool '{tool_name}' must be a dictionary.")
    
    function = TOOL_REGISTRY[tool_name]
    result = function(**arguments)              # Execute the function/tool extracted from openAI LLM response whose type="function_call"
    
    return {"tool_name": tool_name, "result": result, "call_id": call_id}

def validate_and_execute_tool(tool_call: dict) -> dict:
    """ Validate the tool call arguments and execute the corresponding financial tool. """
    
    validated_tool_call = validate_tool_arguments(tool_call)
    return execute_tool(validated_tool_call)


def build_tool_outputs(tool_results: list[dict]) -> list[dict]:
    """Convert executed tool results into Responses API input items."""
    
    outputs = []
    
    for tool_result in tool_results:
        call_id = tool_result.get("call_id")
        
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("Tool result is missing a valid call ID.")
    
        if "result" not in tool_result:
            raise ValueError("Tool result is missing the result value.")
        
        outputs.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(tool_result["result"]),
        })
        
    return outputs

def send_tool_outputs(previous_response_id: str, tool_results: list[dict]) -> None: 
    """
    Send the executed tool results back to OpenAI.

    Args:
        previous_response_id: The ID of the previous OpenAI response.
        tool_results: List of executed tool results.
    """

    if not previous_response_id:
        raise ValueError("Previous response ID is required.")
    
    outputs = build_tool_outputs(tool_results)
    if not outputs:
        raise ValueError("At least one tool result is required.")
    
    client = create_openai_client()
    model = get_openai_model()

    response = client.responses.create(
        model=model,
        instructions = SYSTEM_INSTRUCTIONS,
        previous_response_id=previous_response_id,
        input=outputs,                                      # Send executed tool results back to OpenAI as "inputs" for further processing and response generation.
        tools=get_openai_tools(),
        tool_choice="auto",
    )

    return response

def generate_final_response(response) ->str:
    """
    Extract and return the final user-friendly response from OpenAI.
    """
    if response.status != "completed":
        raise RuntimeError(
            f"OpenAI response is not completed. Status: {response.status}"
        )

    final_answer = response.output_text

    if not isinstance(final_answer, str) or not final_answer.strip():
        raise RuntimeError("OpenAI returned an empty final response.")

    # Remove Markdown bold and italic markers
    final_answer = re.sub(r"\*{1,3}", "", final_answer)

    return final_answer.strip()

def run_agent(question: str, max_iterations: int = 5) -> str:
    """
    Run the financial agent until OpenAI returns a final response or the maximum number of iterations is reached.
    This supports multiple tool calls and multiple tool calling rounds.
    """
    
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    
    # Send the initial question to OpenAI
    response = send_financial_question(question)
    
    iteration = 0
    while True:
        # Detect function calls in the OpenAI response
        tool_calls = extract_tool_calls(response)

        # If no tool calls are detected, return the final response
        if not tool_calls:
            return generate_final_response(response)
        
        # Prevent infinite tool-calling loops
        if iteration >= max_iterations:
            raise RuntimeError(
                f"Agent exceeded maximum tool-calling rounds: "
                f"{max_iterations}"
            )
        print(f"\nTool-calling round: {iteration + 1}")
        
        # Validate and execute each tool call
        tool_results = []
        for tool_call in tool_calls:
            print(f"\nExecuting tool: {tool_call['tool_name']}")
            result = validate_and_execute_tool(tool_call)
            print("Tool result:", result['result'])
            tool_results.append(result)
        
        # Send all tool results back to OpenAI  
        response = send_tool_outputs(
            previous_response_id=response.id,
            tool_results=tool_results
        )
        iteration += 1
        

def main() -> None:
    """Run a sample financial question."""

    question = input(
        "\nEnter your financial question: "
    ).strip()

    if not question:
        print("Please enter a valid question.")
        return

    try:
        final_answer = run_agent(question)       

        print("\n========== FINAL ANSWER ==========")
        print(f"\nFinancial Assistant:\n{final_answer}")
        print("==================================")

    except Exception as exc:
        print(f"\nAgent error: {exc}")


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