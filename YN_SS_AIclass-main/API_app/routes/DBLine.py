from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status

from API_app.model.dataclass import DataInput, PredictOutput

from sqlalchemy.orm import Session
from DW.DB.connection import conn_db, get_session
from DW.DB.query import crud
# from DW.DB.db_models import test_model
# from apis import test # main logic
from DW.DB.connection import get_session
from DW.DB.query import crud
from DW.DB.db_models import base_model
from API_app.apis import test # main logic


conDB = APIRouter(
    prefix="/conDB",
    tags=['conDB'],# url 앞에 고정적으로 붙는 prefix path 추가
) # root에서 분리된 경로

@conDB.get("/", status_code=status.HTTP_200_OK) # Route Path
def test_index(db: Session = Depends(get_session)):
    res = crud.get_items(db=db) # apis 호출
    if res is None:
        raise HTTPException(status_code=404, detail="Id not found")
    return {"res" : res,} # 결과

@conDB.get("/read/{id}", status_code=status.HTTP_200_OK)
def read_Item(id,db:Session=Depends(get_session)):
    res = crud.get_item(id,db) #db.query(test_model.Test_Table).all()
    if res is None:
        raise HTTPException(status_code=404, detail="Id not found")
    return {"res":res} 

# @conDB.post("/post")
# def create_Item(Input:DataInput,db:Session=Depends(get_session)):
#     DB_Table = test_model.Test_Table()
    
#     db.add(DB_Table)
#     db.commit()
#     return Input

# @conDB.put("/")
# def update_Item

# @conDB.delete("/")
# def delete_Item():
    