from openai import OpenAIError

from app.openai_client import (
    create_openai_client,
    get_openai_model,
)

from app.tool_definitions import get_openai_tools


SYSTEM_INSTRUCTIONS = """
You are a financial calculation assistant.

You have access to financial calculation tools.

Use the available tools when the user requests a financial
calculation.

Do not invent calculation results.

The application will execute the requested tools separately.
"""


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
    tools = get_openai_tools()

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

    function_calls = get_function_calls(response)

    if function_calls:

        print("\nOpenAI requested the following tool(s):")

        for call in function_calls:

            print("\nTool name:", call.name)
            print("Call ID:", call.call_id)
            print("Raw arguments:", call.arguments)

    else:

        output_text = getattr(response, "output_text", None)

        if output_text:
            print("\nOpenAI response:")
            print(output_text)

        else:
            print("\nNo function call or text response was returned.")


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


if __name__ == "__main__":
    main()