from app.tool_definitions import build_function_tool, get_openai_tools

def test_three_tools_are_defined():
	tools = get_openai_tools()

	assert len(tools) == 3
	assert tools[0]["name"] == "calculate_roi"
	assert tools[1]["name"] == "calculate_profit"
	assert tools[2]["name"] == "calculate_compound_interest"
	
def test_all_tools_use_responses_api_format():
	tools = get_openai_tools()
	for tool in tools:
		assert tool["type"] == "function"
		assert "name" in tool
		assert "description" in tool
		assert "parameters" in tool
		assert "strict" in tool
		
def test_all_tool_schemas_have_required_properties():
	tools = get_openai_tools()
	for tool in tools:
		parameters = tool["parameters"]
		assert "properties" in parameters
		assert "required" in parameters
		assert isinstance(parameters["required"], list)
		assert set(parameters["required"]) == set(parameters["properties"].keys())
		
def test_roi_tool_contains_correct_validation_rules():
	tools = get_openai_tools()
	roi_tool = next(tool for tool in tools if tool["name"] == "calculate_roi")
	parameters = roi_tool["parameters"]
	properties = parameters["properties"]
	
	assert "investment" in properties
	assert "profit" in properties
	
	investment_schema = properties["investment"]
	profit_schema = properties["profit"]
	
	assert investment_schema["type"] == "number"
	assert profit_schema["type"] == "number"
	

def test_profit_tool_contains_correct_validation_rules():
    """Profit tool should reject negative revenue and cost."""

    tools = get_openai_tools()

    profit_tool = next(
        tool for tool in tools
        if tool["name"] == "calculate_profit"
    )

    properties = profit_tool["parameters"]["properties"]

    assert properties["revenue"]["minimum"] == 0
    assert properties["cost"]["minimum"] == 0


def test_compound_interest_tool_contains_correct_rules():
    """Compound-interest tool should expose valid constraints."""

    tools = get_openai_tools()

    interest_tool = next(
        tool for tool in tools
        if tool["name"] == "calculate_compound_interest"
    )

    properties = interest_tool["parameters"]["properties"]

    assert properties["principal"]["minimum"] == 0
    assert properties["annual_rate"]["minimum"] == -100
    assert properties["years"]["minimum"] == 0
	