
BASE_URL = {
                "ECOS"  :'http://ecos.bok.or.kr/api',
                "KOSIS" :'https://kosis.kr/openapi'
            }

SERVICE_DICT = {
        "ECOS" :{
        # 서비스명에 대한 Key 값의 변경 필요
                    "서비스통계목록" : {
                            "service_name" : "StatisticTableList",
                            "service_param": ["service","API_KEY","format","language","start_nmb","end_nmb","stats_code"]
                        },
                    "통계용어사전" : {
                            "service_name" : "StatisticWord",
                            "service_param": ["service","API_KEY","format","language","start_nmb","end_nmb","Terms"]
                        },
                    "통계세부항목목록" : {
                            "service_name" : "StatisticItemList",
                            "service_param": ["service","API_KEY","format","language","start_nmb","end_nmb","stats_code"] 
                        },
                    "100대통계지표" : {
                            "service_name" : "KeyStatisticList",
                            "service_param": ["service","API_KEY","format","language","start_nmb","end_nmb"]
                        },
                    "통계조회조건설정" : {
                            "service_name" : "StatisticSearch",
                            "service_param": ["service","API_KEY","format","language","start_nmb","end_nmb","stats_code",
                                              "Cycle","Start_date","End_date","Item_code_1","Item_code_2","Item_code_3",
                                              "Item_code_4"]
                        },
                    "통계메타DB" : {
                            "service_name" : "StatisticMeta",
                            "service_param": ["service","API_KEY","format","language","start_nmb","end_nmb","Meta_data"]
                        }
                },
        
        "KOSIS": {
                    "KOSIS통합검색" : {
                            "service_name": "statisticsSearch.do?method=getList",
                            "service_param": ["API_KEY","searchNm","orgId","sort","startCount","resultCount","format","jsonVD","jsonMVD"], # input : "searchNm","sort","startCount","resultCount"
                            "columns": ['ORG_ID', 'ORG_NM', 'TBL_ID', 'TBL_NM', 'STAT_ID', 'STAT_NM', 'VW_CD', 'MT_ATITLE',
                                        'FULL_PATH_ID', 'CONTENTS', 'STRT_PRD_DE', 'END_PRD_DE', 'ITEM03', 'REC_TBL_SE',
                                        'TBL_VIEW_URL', 'LINK_URL', 'STAT_DB_CNT', 'QUERY'],
                        },
                    "통계설명" : {
                            "service_name": "statisticsExplData.do?method=getList",  #  &statId=통계조사ID 또는  &orgId=기관ID&tblId=통계표ID
                            "service_param": ["API_KEY","statId","metaItm","format","jsonVD","jsonMVD"],    # input : 
                            "columns": ['statsNm', 'statsKind', 'statsContinue', 'basisLaw', 'writingPurps', 'statsPeriod',
                                        'writingSystem', 'pubExtent', 'pubPeriod', 'writingTel', 'statsField', 'examinObjrange',
                                        'examinObjArea', 'josaUnit', 'applyGroup', 'josaItm','pubPeriod', 'pubExtent',
                                        'publictMth', 'examinTrgetPd', 'dataUserNote', 'mainTermExpl', 'dataCollectMth', 'examinHistory', 'confmNo',
                                        'confmDt', 'statsEnd'],                            
                        },
                    "통계표설명": {
                            "service_name" : "statisticsData.do?method=getMeta",
                            "sub_service_name" : {
                                        "통계표명칭": {
                                                     "type"   : "&type=TBL",
                                                     "service_param"  : ["API_KEY","orgId","tblId","format"], # input : "orgId","tblId"
                                                     "columns": ['TBL_NM', 'TBL_NM_ENG',]
                                                     },
                                        "기관명칭": { 
                                                     "type"   : "&type=ORG",
                                                     "service_param"  : ["API_KEY","orgId","format"],         # input : "orgId"
                                                     "columns": ['ORG_NM', 'ORG_NM_ENG',]
                                                    },
                                        "수록정보": { 
                                                     "type"   : "&type=PRD",
                                                     "service_param"  : ["API_KEY","orgId","tblId","format","detail"],  # input : "orgId","tblId","detail"
                                                     "columns": ['PRD_SE', 'PRD_DE',]
                                                    },
                                        "분류항목": { 
                                                     "type"   : "&type=ITM",
                                                     "service_param"  : ["API_KEY","orgId","tblId","objId","itmId","format"],  # input : "orgId","tblId","objId","itmId"
                                                     "columns": ['OBJ_ID', 'OBJ_NM', 'OBJ_NM_ENG', 'ITM_ID', 'ITM_NM', 'ITM_NM_ENG', 
                                                                 'UP_ITM_ID', 'OBJ_ID_SN', 'UNIT_ID', 'UNIT_NM', 'UNIT_ENG_NM']
                                                    },
                                        "주석": { 
                                                     "type"   : "&type=CMMT",
                                                     "service_param"  : ["API_KEY","orgId","tblId","format"],  # input : "orgId","tblId"
                                                     "columns": ['CMMT_NM', 'CMMT_DC', 'OBJ_ID', 'OBJ_NM', 'ITM_ID', 'ITM_NM']
                                                    },
                                        "단위": { 
                                                     "type"   : "&type=UNIT",
                                                     "service_param"  : ["API_KEY","orgId","tblId","format"],  # input : "orgId","tblId"
                                                     "columns": ["UNIT_NM", "UNIT_NM_ENG"]
                                                    },
                                        "출처": { 
                                                     "type"   : "&type=SOURCE",
                                                     "service_param"  : ["API_KEY","orgId","tblId","format"],  # input : "orgId","tblId"
                                                     "columns": ["JOSA_NM", "DEPT_NM", "DEPT_PHONE", "STAT_ID"]
                                                    },
                                        "가중치": { 
                                                     "type"   : "&type=WGT",
                                                     "service_param"  : ["API_KEY","orgId","tblId","C1","C2","C3","C4","C5","C6","C7","C8","ITEM","format"], # input : "orgId","tblId","C1~C8" ,"ITEM"
                                                     "columns": ['C1', 'C1_NM', 'C2', 'C2_NM', 'C3', 'C3_NM', 'C4', 'C4_NM', 'C5', 'C5_NM',
                                                                 'C6', 'C6_NM', 'C7', 'C7_NM', 'C8', 'C8_NM', 'ITM_ID', 'ITM_NM', 'WGT_CO']
                                                    },
                                        "자료갱신일": { 
                                                     "type"   : "&type=NCD",
                                                     "service_param"  : ["API_KEY","orgId","tblId","prdSe","format"],  # input : "orgId","tblId","prdSe"  // ?? "jsonVD","jsonMVD"
                                                     "columns": ['ORG_NM', 'TBL_NM', 'PRD_SE', 'PRD_DE', 'SEND_DE',]
                                                    },
                                        },
                        },
                    "통계목록": {
                            "service_name" : "statisticsList.do?method=getList",
                            "service_param": ["API_KEY","vwCd","parentListId","format","jsonVD","jsonMVD"],             # input : "vwCd","parentListId"
                            "columns": ['VW_CD', 'VW_NM', 'LIST_ID', 'LIST_NM', 'ORG_ID', 'TBL_ID', 'TBL_NM','STAT_ID','SEND_DE','REC_TBL_SE']
                        },
                    
                    "통계자료": {
                            "service_name" : "Param/statisticsParameterData.do?method=getList",
                            "service_param": ["API_KEY","orgId","tblId","objL1","objL2","objL3","objL4","objL5","objL6","objL7","objL8",
                                      "itmId","prdSe","startPrdDe","endPrdDe","newEstPrdCnt","prdInterval","format","outputFields","jsonVD","jsonMVD"], # input : "userStatsId","prdSe","startPrdDe","endPrdDe","newEstPrdCnt","prdInterval",
                            "columns": ['ORG_ID', 'TBL_ID', 'TBL_NM',
                                        'C1_OBJ_NM', 'C1_OBJ_NM_ENG', 'C1_NM', 'C1_NM_ENG','C1',  
                                        'C2_OBJ_NM', 'C2_OBJ_NM_ENG', 'C2_NM', 'C2_NM_ENG','C2', 
                                        'C3_OBJ_NM', 'C3_OBJ_NM_ENG', 'C3_NM', 'C3_NM_ENG','C3', 
                                        'C4_OBJ_NM', 'C4_OBJ_NM_ENG', 'C4_NM', 'C4_NM_ENG','C4',  
                                        'C5_OBJ_NM', 'C5_OBJ_NM_ENG', 'C5_NM', 'C5_NM_ENG','C5', 
                                        'C6_OBJ_NM', 'C6_OBJ_NM_ENG', 'C6_NM', 'C6_NM_ENG','C6', 
                                        'C7_OBJ_NM', 'C7_OBJ_NM_ENG', 'C7_NM', 'C7_NM_ENG','C7',  
                                        'C8_OBJ_NM', 'C8_OBJ_NM_ENG', 'C8_NM', 'C8_NM_ENG','C8', 
                                        'ITM_ID', 'ITM_NM', 'ITM_NM_ENG','UNIT_ID', 'UNIT_NM', 'UNIT_NM_ENG', 'PRD_SE', 'PRD_DE', 'DT',]
                        },
        },
    }

TRANS_DICT = {
    "ECOS" :{
            'P_STAT_CODE': '상위통계표코드',
            'STAT_CODE': '통계표코드',
            'STAT_NAME': '통계명',
            'CYCLE': '시점',
            'SRCH_YN': '검색가능여부',
            'ORG_NAME': '출처',
            'WORD': '용어',
            'CONTENT': '용어설명',
            'GRP_CODE': '항목그룹코드',
            'GRP_NAME': '항목그룹명',
            'ITEM_CODE': '통계항목코드',
            'ITEM_NAME': '통계항목명',
            'P_ITEM_CODE': '상위통계항목코드',
            'P_ITEM_NAME': '상위통계항목명',
            'START_TIME': '수록시작일자',
            'END_TIME': '수록종료일자',
            'DATA_CNT': '자료수',
            'UNIT_NAME': '단위',
            'WEIGHT': '가중치',
            'ITEM_CODE1': '통계항목코드1',
            'ITEM_NAME1': '통계항목명1',
            'ITEM_CODE2': '통계항목코드2',
            'ITEM_NAME2': '통계항목명2',
            'ITEM_CODE3': '통계항목코드3',
            'ITEM_NAME3': '통계항목명3',
            'ITEM_CODE4': '통계항목코드4',
            'ITEM_NAME4': '통계항목명4',
            'TIME': '시점',
            'DATA_VALUE': '값',
            'CLASS_NAME': '통계그룹명',
            'KEYSTAT_NAME': '통계명',
            'LVL': '레벨',
            'P_CONT_CODE': '상위통계항목코드',
            'CONT_CODE': '통계항목코드',
            'CONT_NAME': '통계항목명',
            'META_DATA': '메타데이터'
        },
    
    "KOSIS" :{
            "KOSIS통합검색": {
                'ORG_ID': '기관ID',
                'ORG_NM': '기관명',
                'TBL_ID': '통계표ID',
                'TBL_NM': '통계표명',
                'STAT_ID': '조사ID',
                'STAT_NM': '조사명',
                'VW_CD': 'KOSIS목록구분',
                'MT_ATITLE': 'KOSIS통계표위치',
                'FULL_PATH_ID': '통계표위치',
                'CONTENTS': '통계표주요내용',
                'STRT_PRD_DE': '수록기간시작일',
                'END_PRD_DE': '수록기간종료일',
                'ITEM03': '통계표주석',
                'REC_TBL_SE': '추천통계표여부',
                'TBL_VIEW_URL': 'KOSIS목록URL',
                'LINK_URL': 'KOSIS통계표URL',
                'STAT_DB_CNT': '검색결과건수',
                'QUERY': '검색어명',
            },
            "통계설명": {
                'statsNm': '조사명',
                'statsKind': '통계종류',
                'statsContinue': '계속여부',
                'basisLaw': '법적근거',
                'writingPurps': '조사목적',
                'statsPeriod': '조사주기',
                'writingSystem': '조사체계',
                'pubExtent': '공표범위',
                'pubPeriod': '공표주기',
                'writingTel': '연락처',
                'statsField': '통계활용분야실태',
                'examinObjrange': '조사대상범위',
                'examinObjArea': '조사대상지역',
                'josaUnit': '조사단위및조사대상규모',
                'applyGroup': '적용분류',
                'josaItm': '조사항목',
                'publictMth': '공표방법및URL',
                'examinTrgetPd': '조사대상기간및조사기준시점',
                'examinPd': '조사기간',
                'dataUserNote': '자료이용자유의사항',
                'mainTermExpl': '주요용어해설',
                'dataCollectMth': '자료수집방법',
                'examinHistory': '조사연혁',
                'confmNo': '승인번호',
                'confmDt': '승인일자',
                'statsEnd': '통계종료',
            },
            "통계표설명": {
                "통계표명칭": {
                    'TBL_NM': '통계표명',
                    'TBL_NM_ENG': '통계표영문명',
                },
                "기관명칭": {
                    'ORG_NM': '기관명',
                    'ORG_NM_ENG': '기관영문명',
                },
                "수록정보": {
                    'PRD_SE': '수록주기',
                    'PRD_DE': '수록시점',
                },
                "분류항목": {
                    'ORG_ID': '기관ID',
                    'TBL_ID': '통계표ID',
                    'CD_ID': '코드ID',
                    'CD_NM': '코드명',
                    'OBJ_ID': '분류ID',
                    'OBJ_NM': '분류명',
                    'OBJ_NM_ENG': '분류영문명',
                    'ITM_ID': '분류값ID',
                    'ITM_NM': '분류값명',
                    'ITM_NM_ENG': '분류값영문명',
                    'UP_ITM_ID': '상위분류값ID',
                    'OBJ_ID_SN': '분류값순번',
                    'UNIT_ID': '단위ID',
                    'UNIT_NM': '단위명',
                    'UNIT_ENG_NM': '단위영문명',
                },
                "주석": {
                    'CMMT_NM': '주석유형',
                    'CMMT_DC': '주석',
                    'OBJ_ID': '분류ID',
                    'OBJ_NM': '분류명',
                    'ITM_ID': '분류값ID',
                    'ITM_NM': '분류값명',
                },
                "단위": {
                    'UNIT_NM': '단위명',
                    'UNIT_NM_ENG': '단위영문명',
                },
                "출처": {
                    'JOSA_NM': '조사명',
                    'DEPT_NM': '통계표담당부서',
                    'DEPT_PHONE': '통계표담당부서전화번호',
                },
                "가중치": {
                    'C1': '분류값ID1',
                    'C1_NM': '분류값명1',
                    'C2': '분류값ID2',
                    'C2_NM': '분류값명2',
                    'C3': '분류값ID3',
                    'C3_NM': '분류값명3',
                    'C4': '분류값ID4',
                    'C4_NM': '분류값명4',
                    'C5': '분류값ID5',
                    'C5_NM': '분류값명5',
                    'C6': '분류값ID6',
                    'C6_NM': '분류값명6',
                    'C7': '분류값ID7',
                    'C7_NM': '분류값명7',
                    'C8': '분류값ID8',
                    'C8_NM': '분류값명8',
                    'ITM_ID': '항목ID',
                    'ITM_NM': '항목명',
                    'WGT_CO': '가중치',
                },
                "자료갱신일": {
                    'ORG_NM': '기관명',
                    'TBL_NM': '통계표명',
                    'PRD_SE': '수록주기',
                    'PRD_DE': '수록시점',
                    'SEND_DE': '자료갱신일',
                },
            },
            "통계목록": {
                'VW_CD': '서비스뷰ID',
                'VW_NM': '서비스뷰명',
                'LIST_ID': '목록ID',
                'LIST_NM': '목록명',
                'ORG_ID': '기관ID',
                'TBL_ID': '통계표ID',
                'TBL_NM': '통계표명',
                'REC_TBL_SE': '추천통계표여부',
            },
            "통계자료": {
                'ORG_ID': '기관ID',
                'TBL_ID': '통계표ID',
                'TBL_NM': '통계표명',
                'C1_OBJ_NM': '분류명1',
                'C1_OBJ_NM_ENG': '분류영문명1',
                'C1_NM': '분류값명1',
                'C1_NM_ENG': '분류값영문명1',
                'C1': '분류값ID1',
                'C2_OBJ_NM': '분류명2',
                'C2_OBJ_NM_ENG': '분류영문명2',
                'C2_NM': '분류값명2',
                'C2_NM_ENG': '분류값영문명2',
                'C2': '분류값ID2',
                'C3_OBJ_NM': '분류명3',
                'C3_OBJ_NM_ENG': '분류영문명3',
                'C3_NM': '분류값명3',
                'C3_NM_ENG': '분류값영문명3',
                'C3': '분류값ID3',
                'C4_OBJ_NM': '분류명4',
                'C4_OBJ_NM_ENG': '분류영문명4',
                'C4_NM': '분류값명4',
                'C4_NM_ENG': '분류값영문명4',
                'C4': '분류값ID4',
                'C5_OBJ_NM': '분류명5',
                'C5_OBJ_NM_ENG': '분류영문명5',
                'C5_NM': '분류값명5',
                'C5_NM_ENG': '분류값영문명5',
                'C5': '분류값ID5',
                'C6_OBJ_NM': '분류명6',
                'C6_OBJ_NM_ENG': '분류영문명6',
                'C6_NM': '분류값명6',
                'C6_NM_ENG': '분류값영문명6',
                'C6': '분류값ID6',
                'C7_OBJ_NM': '분류명7',
                'C7_OBJ_NM_ENG': '분류영문명7',
                'C7_NM': '분류값명7',
                'C7_NM_ENG': '분류값영문명7',
                'C7': '분류값ID7',
                'C8_OBJ_NM': '분류명8',
                'C8_OBJ_NM_ENG': '분류영문명8',
                'C8_NM': '분류값명8',
                'C8_NM_ENG': '분류값영문명8',
                'C8': '분류값ID8',
                'ITM_ID': '항목ID',
                'ITM_NM': '항목명',
                'ITM_NM_ENG': '항목영문명',
                'UNIT_ID': '단위ID',
                'UNIT_NM': '단위명',
                'UNIT_NM_ENG': '단위영문명',
                'PRD_SE': '수록주기',
                'PRD_DE': '수록시점',
                'DT': '수치값',
            },
        }
    }
