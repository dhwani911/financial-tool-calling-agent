from pydantic import BaseModel, Field, ConfigDict

class ROIInput(BaseModel):
	""" Input parameters for calculation of ROIs """
	
	model_config = ConfigDict(extra='forbid', allow_inf_nan=False)   # "forbid" rejects unexpected fields instead of silently accepting them.
	
	investment: float = Field(gt=0, description='The initial amount invested. Must be greater than zero.')
	profit: float = Field(description='The profit earned from the investment. May be negative.')

class ProfitInput(BaseModel):
	"""Validate inputs for calculation of Profit"""

	model_config = ConfigDict(extra='forbid',allow_inf_nan=False) # In Pydantic, allow_inf_nan=False is a validation setting that rejects infinite values (inf, -inf)

	revenue: float = Field(ge=0, description="Total revenue. Must be zero or greater.")
	cost: float = Field(ge=0, description="Total cost. Must be zero or greater.")


class CompoundInterestInput(BaseModel):
    """Validate inputs for calculating compound interest."""
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)

    principal: float = Field(ge=0, description="Initial principal amount. Must be zero or greater.")
    annual_rate: float = Field(ge=-100, description="Annual interest rate as a percentage. Must be at least -100.")
    years: int = Field(ge=0, description="Investment duration in years. Must be zero or greater.")