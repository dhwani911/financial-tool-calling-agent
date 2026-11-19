import json

from openai import OpenAIError
from app.openai_client import create_openai_client, get_openai_model
from app.tool_definitions import get_openai_tools


SYSTEM_INSTRUCTIONS = """
You are a financial calculation assistant.

You have access to financial calculation tools.

Use the available tools when the user requests a financial calculation.

Do not invent calculation results.

The application will execute the requested tools separately.
"""

def send_financial_question(question: str) -> dict:
	"""
	Send a financial question and available tools to OpenAI.

    This function only requests and inspects the model response.
    It does not execute the requested Python financial function.

	Args:
		question (str): The financial question to ask.
	"""

	if not isinstance(question, str) or not question.strip():
		raise ValueError("Question must be a non-empty string.")
	
	# Initialize the OpenAI client 
	client = create_openai_client()
	
	# Retrieve the model name.
	model = get_openai_model()
	
	# Load available Finance tools.
	tools = get_openai_tools()         # All the financial calculation tools available for the model to use.
	
	# Send the question and tool definitions to OpenAI.
	response = client.responses.create(
		model=model,
		instructions=SYSTEM_INSTRUCTIONS,
		input=question.strip(),
		tools=tools,
		tool_choice="required",
	)
	
	return response

def display_response(response: dict) -> None:
	"""
	Display the model's text response or requested function calls.

	Args:
		response (dict): The response from the OpenAI API.
	"""

	print("\nResponse ID:", response.id)
	print("Response status:", response.status)
	
	function_calls = [item for item in response.output if item.type == "function_call"]
	
	if function_calls:
		print("\nOpenAI requested the following tool(s):")
		for call in function_calls:
			print(f"\n Tool name - {call.name}")
			print("Call Id:", call.id)
			
			try:
				arguments = json.loads(call.arguments)

				print(
					"Arguments:",
					json.dumps(arguments, indent=4),
				)

			except json.JSONDecodeError:
				print("Raw arguments:", call.arguments)

	elif response.output_text:
		print("\nOpenAI response:")
		print(response.output_text)

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