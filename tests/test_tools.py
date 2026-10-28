import pytest

from app.tools import calculate_roi, calculate_profit, calculate_compound_interest

def test_calculate_roi():
	investment, profit = 10000, 2500
	result = calculate_roi(investment, profit)
	assert result == pytest.approx(25.0)

def test_calculate_profit():
	revenue, cost = 10000, 7000
	result = calculate_profit(revenue, cost)
	assert result == pytest.approx(3000)

def test_calculate_compound_interest():
	principal, annual_rate, years = 1000, 10, 2 
	result = calculate_compound_interest(principal, annual_rate, years)
	assert result == pytest.approx(1210.0)
	
# ROI Edge cases

@pytest.mark.parametrize("investment",[0,-10000])
def test_roi_rejects_non_positive_investment(investment):
	with pytest.raises(ValueError):
		calculate_roi(investment=investment, profit=10000)
	
def test_roi_allows_negative_profit():
	result = calculate_roi(investment=10000, profit=-4000)
	assert result == pytest.approx(-40.0)
	
@pytest.mark.parametrize("invalid_value",[float("nan"), float("inf"), float("-inf")])
def test_roi_rejects_non_finite_inputs(invalid_value):
	with pytest.raises(ValueError):
		calculate_roi(investment=10000, profit=invalid_value)
		
# Profit Edge cases

def test_profit_with_zero_revenue_and_cost():
	result = calculate_profit(revenue=0, cost=0)
	assert result == 0
	
def test_profit_with_equal_revenue_and_cost():
	result = calculate_profit(revenue=10000, cost=10000)
	assert result == 0
	
@pytest.mark.parametrize("revenue,cost",[(-5000,1000),(10000,-5000)])
def test_profit_rejects_negative_revenue_or_cost(revenue,cost):
	with pytest.raises(ValueError):
		calculate_profit(revenue=revenue,cost=cost)
		
# Compound-interest edge cases

def test_compound_interest_with_zero_principal():
	result = calculate_compound_interest(principal=0, annual_rate=5, years=25)
	assert result == 0
	
def test_compound_interest_with_zero_years():
	result = calculate_compound_interest(principal=20000, annual_rate=5, years=0)
	assert result == 20000

def test_compound_interest_with_zero_rate():
	result = calculate_compound_interest(principal=20000, annual_rate=0, years=25)
	assert result == 20000
	
def test_compound_interest_with_minus_100_percent_rate():
	result = calculate_compound_interest(principal=20000, annual_rate=-100, years=25)
	assert result == 0

def test_compound_interest_rejects_rate_below_minus_100():
	with pytest.raises(ValueError):
	    calculate_compound_interest(principal=20000, annual_rate=-104, years=25)
	    
def test_compound_interest_rejects_negative_principal():
	with pytest.raises(ValueError):
		calculate_compound_interest(principal=-10000, annual_rate=5, years=25)
		
def test_compound_interest_rejects_negative_years():
	with pytest.raises(ValueError):
		calculate_compound_interest(principal=10000, annual_rate=5, years=-3)
		
@pytest.mark.parametrize("rate",[float("nan"), float("inf"), float("-inf")])
def test_compound_interest_rejects_non_finite_rate(rate):
	with pytest.raises(ValueError):
		calculate_compound_interest(principal=10000, annual_rate=rate, years=3)