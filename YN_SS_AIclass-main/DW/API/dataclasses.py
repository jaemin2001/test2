from pydantic import BaseModel

class api_request_param(BaseModel):
    service: str = ""      # SERVICE_DICT.get(1).get('service_name')
    API_KEY: str = ""
    format: str = "json"   # OR XML
    jsonVD: str = "Y"
    jsonMVD: str = "Y"
    language: str = "kr"   # OR en
    start_nmb: str = "1"
    end_nmb: str = "99999" # OR 100
    stats_code: str = ""
    Terms: str = ""
    Meta_data: str =""
    Cycle: str = ""        # 주기 (년:A, 반년:S, 분기:Q, 월:M, 반월:SM, 일: D)
    Start_date: str = ""        # 주기에 맞는 형식으로 입력: 2015, 2015Q1, 201501, 20150101 등
    End_date: str = ""          # 주기에 맞는 형식으로 입력: 2015, 2015Q1, 201501, 20150101 등
    
    Item_code_1: str = ""       # optional, by default None
    Item_code_2: str = ""       # optional, by default None
    Item_code_3: str = ""       # optional, by default None
    Item_code_4: str = ""       # optional, by default None

    searchNm: str = ""	   # 검색명	[필수]
    statId: str = "" 
    metaItm: str = ""
    vwCd: str = ""
    parentListId: str = ""

    orgId: str = ""	           # 기관코드	[선택]
    tblId: str = ""
    objL1: str = ""
    objL2: str = ""
    objL3: str = ""
    objL4: str = ""
    objL5: str = ""
    objL6: str = ""
    objL7: str = ""
    objL8: str = ""
    itmId: str = ""
    C1: str = ""
    C2: str = ""
    C3: str = ""
    C4: str = ""
    C5: str = ""
    C6: str = ""
    C7: str = ""
    C8: str = ""
    ITEM: str = ""
    prdSe: str = ""
    startPrdDe: str = ""
    endPrdDe: str = "" 
    newEstPrdCnt: str = ""
    prdInterval: str = ""
    outputFields: str = ""
    sort: str = ""	           # 정렬 - 비고 : 정확도 RANK, 최신순DATE 
                               # ※ 호출 파라미터에 sort 없을 경우에는 자동으로 RANK 로 정렬	[선택]
    detail: str = ""
    startCount: str	= ""       # 페이지 번호	[선택]
    resultCount: str = ""	   # 데이터 출력 개수 - 비고 : resultCount=20, startCount=1 : 1 ~ 20번 결과 리턴
                               # resultCount=20, startCount=2 : 21 ~ 40번 결과 리턴	[선택]
    
    
    # @classmethod
    # def as_form(cls, any_param: str = Form(...),any_other_param: int = Form(1)):
    #     return cls(any_param=any_param, any_other_param=any_other_param)
