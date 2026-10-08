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