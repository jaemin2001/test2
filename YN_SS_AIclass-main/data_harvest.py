'''
2024.01.20
한국은행 통계시스템 ECOS 자료를 api를 사용하여
데이터를 다운로드하고 원하는 DataFrame 형태로 가공하기 위한 API 접근 라이브러리
'''
import pandas as pd
from DW.API import API_WORK
from DW.API.dataclasses import api_request_param
from DW.API.pipline import pipe
from DW.API.form import SERVICE_DICT
# from UI.DataView import DataViewer

class data_harvest:
    # data_provider의 pipline 클라스를 통해서 필요한 factor를 가져옴.
    def __init__(self, site, service, sub_service) -> None:
        self.__set_host = site
        self.__service = service
        self.__sub_service = sub_service
        self.__pipe = pipe()
        self.__factors = self.__pipe.load_select_factor(self.__service)
        self.__params = api_request_param()
        # self.__result = self.do_it()
                
    def do_it(self):
        i = 1
        df = pd.DataFrame()
        site_res = API_WORK(self.__set_host, params=self.__params)

        for var in self.__factors['Variable']:
            self.__params.stats_code = str(self.__factors[self.__factors['Variable'] == var ]['stats_code'].values[0])
            self.__params.Cycle = str(self.__factors[self.__factors['Variable'] == var ]['Cycle'].values[0])
            self.__params.Start_date = str(self.__factors[self.__factors['Variable'] == var ]['Start_date'].values[0])
            self.__params.End_date = str(self.__factors[self.__factors['Variable'] == var ]['End_date'].values[0])
            self.__params.Item_code_1 = str(self.__factors[self.__factors['Variable'] == var ]['Item_code_1'].values[0])
            self.__params.Item_code_2 = str(self.__factors[self.__factors['Variable'] == var ]['Item_code_2'].values[0])
            
            print(f"num fac : {i}")
            print(f"Variable : {var} , stats_code : {self.__params.stats_code}")
            print(f"Variable : {var} , Cycle : {self.__params.Cycle}")
            print(f"Variable : {var} , Start_date : {self.__params.Start_date}")
            print(f"Variable : {var} , End_date : {self.__params.End_date}")
            i += 1
            site_res.set_params(self.__params)
            res = site_res.api_service(service=self.__service, sub_service=self.__sub_service, trans_colname=False)
            df = pd.concat([df,res[['TIME','DATA_VALUE']]], axis=1)
            
            if i == 10 :
                print(f"respone :\n {df}")
                raise
           
        return df
      
#    def save(self, save_path):
#        pipe.data_save(save_path, self.__result)

def main(site, service, sub_service):  
    # # 현재는 임시로 요청 저장이 있지만 원래 계획은  main() 에는 GUI class를 로드해서 사용할 예정
    params = api_request_param()
    params.service = SERVICE_DICT.get(site).get(service).get("service_name")  
    site_res = API_WORK(site, params=params)
    df = site_res.api_service(service=service, sub_service=sub_service, trans_colname=False)
    df.to_excel('./DW/storage/test_keystats_01.xlsx')
   
    # harvest = data_harvest(site, service, sub_service)
    # result = harvest.do_it()
    # result.to_excel('./DW/Storage/test_search_01.xlsx')

if __name__=="__main__":
    site = input(f"오픈데이터 시스템을 선택하세요. : ")
    service = input(f"원하는 서비스를 입력하세요. : ")
    sub = input(f"하위 서비스를 입력하세요. : ")
    main(str(site), str(service), str(sub))

