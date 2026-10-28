import math

def validate_finite_values(**values:float)->None:
	"""Reject non-numeric, non-finite, or Boolean inputs."""
	
	for name, value in values.items():
		try:
			is_valid = (not isinstance(value,bool) and math.isfinite(value))
		except (TypeError, ValueError):
			is_valid = False
		
		if not is_valid:
			raise ValueError(f"{name} must be a finite numeric value.")
	

def calculate_roi(investment:float, profit:float) -> float:
	"""
	Calculate the Return on Investment (ROI) as a percentage.
	"""
	validate_finite_values(investment=investment, profit=profit)
	
	if investment <= 0:
		raise ValueError("Investment cannot be zero or negative.")
	roi = (profit / investment) * 100
	
	if not math.isfinite(roi):
		raise ValueError("ROI calculation produced an invalid result.")
	
	return roi


def calculate_profit(revenue:float, cost:float) -> float:
	"""
	Calculate the profit based on revenue and cost.
	"""
	validate_finite_values(revenue=revenue, cost=cost)
	if revenue < 0:
		raise ValueError("Revenue must be zero or greater.")
	if cost < 0:
		raise ValueError("Cost should be greater than zero.")
	
	return revenue - cost


def calculate_compound_interest(principal:float, annual_rate:float, years:int) -> float:
	validate_finite_values(principal=principal, annual_rate=annual_rate, years=years)
	
	if principal < 0:
		raise ValueError("Investment cannot be negative.")
	if annual_rate < -100:
		raise ValueError("Annual rate cannot be below -100%")
	if years < 0:
		raise ValueError("Years must be zero or greater")
	
	# Edge cases handling
	
	if principal == 0:
		return 0.0
	if annual_rate == -100:
		return 0.0
	if years == 0:
		return float(principal)
	
	try:
		rate = annual_rate / 100
		amount = principal * ((1 + rate) ** years)
	except OverflowError as exc:
		raise ValueError("Compound-interest result exceeds the supported range.") from exc
	
	if not math.isfinite(amount):
		raise ValueError("Compound-interest calculation produced an invalid result.")
		
	return amount
