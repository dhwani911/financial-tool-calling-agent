def calculate_roi(investment:float, profit:float) -> float:
	"""
	Calculate the Return on Investment (ROI) as a percentage.
	"""
	if investment <= 0:
		raise ValueError("Investment cannot be zero or negative.")
	return (profit / investment) * 100


def calculate_profit(revenue:float, cost:float) -> float:
	"""
	Calculate the profit based on revenue and cost.
	"""
	return revenue - cost


def calculate_compound_interest(principal:float, annual_rate:float, years:int) -> float:
	if principal < 0:
		raise ValueError("Investment cannot be negative.")
	if annual_rate < 0:
		raise ValueError("Annual rate cannot be negative.")
	
	rate = annual_rate / 100
	return principal * ((1 + rate) ** years)