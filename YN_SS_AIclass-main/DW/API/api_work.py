import requests
import pandas as pd
import os
from dotenv import load_dotenv
from .dataclasses import api_request_param
# josn 형태 저장으로 변환??
from .form import BASE_URL, SERVICE_DICT, TRANS_DICT 

class API_WORK:
    __MAX = 3
    __instance = 0
    def __new__(cls, site, params:api_request_param):
        if (cls.__instance > cls.__MAX):
            raise ValueError("Cannot create anymore api connect Object...")
        cls.__instance += 1
        return super().__new__(cls)

    def __init__(self, site, params:api_request_param): 
        load_dotenv(dotenv_path="./core/.env",verbose=True)
        self.__host_site = site
        self.__base_url = BASE_URL.get(site)
        self.__params = params
        self.__params.API_KEY = os.getenv(site)
    
    def set_params(self, set_params):
        self.__params = set_params
    
    def api_service(self, service=None, sub_service=None, trans_colname=True):
        
        match self.__host_site:
            case "ECOS": 
                # 제공 API 서비스에 따라서 data_class 정보 설정        
                self.__params.service = SERVICE_DICT.get(self.__host_site).get(service).get("service_name")
                param_list = SERVICE_DICT.get(self.__host_site).get(service).get("service_param")
                set_params = {k:v for k, v in self.__params.model_dump().items() if k in param_list}
                if self.__params.service == "KeyStatisticList":
                    set_params['end_nmb'] = '100'
                # print(f"{set_params.values()}")
                # raise                  
                set_query_param = '/'.join(list(set_params.values()))
                target_url = f"{self.__base_url}/{set_query_param}"

            case "KOSIS" :
                # 제공 API 서비스에 따라서 data_class 정보 설정
                self.__params.service = SERVICE_DICT.get(self.__host_site).get(service).get("service_name")

                if self.__params.service == "statisticsData.do?method=getMeta":
                    param_list = SERVICE_DICT.get(self.__host_site).get(service).get("sub_service_name").get(sub_service).get("service_param")
                    set_params = {k:v for k, v in self.__params.model_dump().items() if k in param_list}
                    service_url = SERVICE_DICT.get(self.__host_site).get(service).get("service_name")
                    type_url = SERVICE_DICT.get(self.__host_site).get(service).get("sub_service_name").get(sub_service).get("type")
                    target_url = f"{self.__base_url}/{service_url}{type_url}"
                else:
                    param_list = SERVICE_DICT.get(self.__host_site).get(service).get("service_param")
                    set_params = {k:v for k, v in self.__params.model_dump().items() if k in param_list}
                    target_url = f"{self.__base_url}/{SERVICE_DICT.get(self.__host_site).get(service).get("service_name")}"
        
        # API 요청 함수로 보냄.
        df = self.api_request(target_url,set_params,trans_colname)

        return df

    def api_request(self, target_url, set_params, trans_colname=False):
        res = requests.get(target_url, params=set_params, verify=False)
        res_json = res.json()
        if res_json.get("RESULT"):
            if res_json.get("RESULT").get("MESSAGE"):
                print(res_json.get("RESULT").get("MESSAGE"))
                return
        df = pd.DataFrame(res_json[set_params.get('service')]['row'])
        if trans_colname:
            df = self.trans_colname(df)
        return df
    
    def trans_colname(self, df):
        try:
            if self.__host_site == SERVICE_DICT.get(self.__host_site):
                if self.__params.service == "통계표설명" :
                    rename_columns = TRANS_DICT.get(self.__host_site).get(service).get()
                else:
                    rename_columns = TRANS_DICT.get(self.__host_site).get(service)
        except AttributeError:
            raise AttributeError("서비스명을 확인해주세요.")
        
        return df.rename(columns=rename_columns)