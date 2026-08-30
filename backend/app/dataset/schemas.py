from pydantic import BaseModel


class DefectTypeSummary(BaseModel):
    defect_type: str
    count: int


class SplitSummary(BaseModel):
    split: str
    defect_types: list[DefectTypeSummary]


class CategoryDetail(BaseModel):
    category: str
    splits: list[SplitSummary]


class CategoryImages(BaseModel):
    category: str
    split: str
    defect_type: str
    filenames: list[str]
