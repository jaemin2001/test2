from sqlalchemy.orm import Session
from DW.DB.db_models.base_model import Base
from DW.DB.db_models.CSCIDS2017 import CSCIDS2017_BALANCED_ATTK
from sqlalchemy import select

def get_items(db: Session):
    return db.query(Base).all()

def get_item(ind,db:Session):
    return db.query(Base).filter(CSCIDS2017_BALANCED_ATTK.Index.in_([ind])).first()

