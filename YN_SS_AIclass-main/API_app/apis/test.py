from sqlalchemy.orm import Session
from DW.DB.query import crud

def test_index(db):
    something = crud.get_items(db)
    return something