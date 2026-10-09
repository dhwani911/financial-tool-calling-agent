from pydantic import BaseModel, Field, ConfigDict

class ROIInput(BaseModel):
	""" Input parameters for calculation of ROIs """
	
	model_config = ConfigDict(extra='forbid')   # "forbid" rejects unexpected fields instead of silently accepting them.
	
	investment: float = Field(gt=0, description='The initial amount invested. Must be greater than zero.')
	profit: float = Field(description='The profit earned from the investment. May be negative.')