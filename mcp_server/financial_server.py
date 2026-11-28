import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root added to sys.path: {PROJECT_ROOT}")

from mcp.server import MCPServer
from app.tools import calculate_roi as calculate_roi_function

# Create an instance of the MCPServer
mcp = MCPServer("Financial MCP Server")

# Expose the existing ROI calculation function as an MCP tool
@mcp.tool()
def calculate_roi(investment: float, profit: float) -> float:
	"""Calculate Return on Investment (ROI).
	
    Args:
    investment: The original amount invested.
    profit: The profit earned from the investment.

    Returns:
        ROI percentage.
	"""
	return calculate_roi_function(investment, profit)