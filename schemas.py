from typing import Literal

from pydantic import BaseModel, Field


class studentRequestModel(BaseModel):
    """ missing-module-docstring """
    class_name: Literal["ML_in_Prod_1", "ML_in_Prod_2", "Big_Data"]

    stu_name: str = "Mg ba"
    stu_id: int = Field(
        ..., ge=100, le=150, description="Student IDs must between 100 and 150"
    )

    stu_age: int = Field(
        ..., ge=16, le=35, description="Student IDs must between 100 and 150"
    )


class textRequestModel(BaseModel):
    """ missing-module-docstring """
    prompt: str = "What is deep learning"


class textResponseModel(BaseModel):
    """ missing-module-docstring """
    execution_time: int = 0
    result: str = ""
