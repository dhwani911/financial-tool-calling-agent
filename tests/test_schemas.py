from locale import currency

import pytest
from app.schemas import ROIInput
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