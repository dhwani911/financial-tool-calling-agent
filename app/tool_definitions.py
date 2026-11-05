from typing import Any
from pydantic import BaseModel

from app.schemas import ROIInput, ProfitInput, CompoundInterestInput

def build_function_tool(name: str, description: str, input_schema: BaseModel) -> dict[str, Any]:
	"""
	Build a API Function tool definition for a function.

	Args:
		name (str): The name of the tool.
		description (str): A brief description of the tool's functionality.
		input_schema (BaseModel): A Pydantic model defining the expected input parameters.

	Returns:
		dict[str, Any]: A dictionary representing the tool definition.
	"""
	parameters = input_schema.model_json_schema()  # Generate the JSON schema for the input parameters

	# Remove titles to keep the JSON schema concise.
	parameters.pop("title", None)
	
	for property_schema in parameters.get("properties", {}).values():
		property_schema.pop("title", None)
		
	return {
		"type": "function",
		"name": name,
		"description": description,
		"parameters": parameters,
		"strict": True,  # Enforce strict validation of input parameters
	}

def get_openai_tools() -> list[dict[str, Any]]:
	"""
	Get a list of OpenAI tool definitions for the available functions.

	Returns:
		list[dict[str, Any]]: A list of tool definitions.
	"""
	return [
		build_function_tool(
			name="calculate_roi",
			description="Calculate the Return on Investment (ROI) based on the initial investment and profit.",
			input_schema=ROIInput,
		),
		build_function_tool(
			name="calculate_profit",
			description="Calculate the Profit based on revenue and cost.",
			input_schema=ProfitInput,
		),
		build_function_tool(
			name="calculate_compound_interest",
			description="Calculate the Compound Interest based on principal, annual rate, and years.",
			input_schema=CompoundInterestInput,
		),
	]

