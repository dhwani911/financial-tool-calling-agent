import pytest
from pygments.lexers import data

from app.schemas import ROIInput, ProfitInput, CompoundInterestInput
from pydantic import ValidationError

def test_valid_ROI_input():
	data = ROIInput(investment=1000.0, profit=500.0)
	assert data.investment == 1000.0
	assert data.profit == 500.0

def test_negative_investment_is_rejected():
	with pytest.raises(ValidationError):
		ROIInput(investment=-1000.0, profit=500.0)
		
def test_unexpected_field_is_rejected():
	with pytest.raises(ValidationError):
		ROIInput(investment=1000.0, profit=500.0, currency='CAD')     # 'currency' is not defined in the schema, should raise ValidationError
		
def test_valid_Profit_input():
	data = ProfitInput(revenue=20000, cost=8000)
	assert data.revenue == 20000
	assert data.cost == 8000
	
def test_profit_rejects_negative_revenue():
	with pytest.raises(ValidationError):
		ProfitInput(revenue=-20000, cost=8000)
		
def test_profit_rejects_negative_cost():
	with pytest.raises(ValidationError):
		ProfitInput(revenue=10000, cost=-8000)
		
def test_valid_compound_interest_input():
	data = CompoundInterestInput(principal=20000, annual_rate=5, years=25)
	assert data.principal == 20000
	assert data.years == 25
	
def test_compound_interest_rejects_negative_principal():
	with pytest.raises(ValidationError):
		CompoundInterestInput(principal=-20000, annual_rate=5, years=25)
		
def test_compound_interest_rejects_rate_below_minus_100():
	with pytest.raises(ValidationError):
		CompoundInterestInput(principal=20000, annual_rate=-101, years=25)
		
def test_compound_interest_rejects_negative_years():
	with pytest.raises(ValidationError):
		CompoundInterestInput(principal=20000, annual_rate=5, years=-10)
		
def test_roi_rejects_zero_investment():
	with pytest.raises(ValidationError):
		ROIInput(investment=0, profit=1000)

def test_roi_allows_negative_profit():
	result = ROIInput(investment=10000, profit=-3000)
	assert result.profit == -3000
	
def test_profit_accepts_zero_revenue_and_cost():
	data = ProfitInput(revenue=0, cost=0)
	assert data.revenue == 0
	assert data.cost == 0
	
def test_compound_interest_accepts_zero_values():
	data = CompoundInterestInput(principal=0, annual_rate=0, years=0)
	assert data.principal == 0, data.years == 0
	
def test_compound_interest_accepts_minus_100_percent_rate():
	data = CompoundInterestInput(principal=10000, annual_rate=-100, years=25)
	assert data.annual_rate == -100
	
@pytest.mark.parametrize(
    "schema, valid_data, numeric_fields",
    [
        (
            ROIInput,
            {"investment": 10000, "profit": 2500},
            ("investment", "profit"),
        ),
        (
            ProfitInput,
            {"revenue": 20000, "cost": 12000},
            ("revenue", "cost"),
        ),
        (
            CompoundInterestInput,
            {
                "principal": 10000,
                "annual_rate": 5,
                "years": 3,
            },
            ("principal", "annual_rate", "years"),
        ),
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_schemas_reject_non_finite_values(
    schema,
    valid_data,
    numeric_fields,
    invalid_value,
):
    """Every numeric field must reject NaN and infinity."""

    for field in numeric_fields:
        test_data = valid_data.copy()
        test_data[field] = invalid_value

        with pytest.raises(ValidationError):
            schema(**test_data)