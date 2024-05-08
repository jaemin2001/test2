from typing import List
from typing import Optional
from sqlalchemy import Column, Float, String, Integer, ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base 
from .base_model import Base

class CSCIDS2017_BALANCED_ATTK(Base):
    __tablename__ = "CSCIDS2017_BALANCED_ATTK"
    
    Index = Column(Integer, primary_key=True)
    FLOW_ID = Column(Float)
    SOURCE_IP = Column(Float)
    SOURCE_PORT = Column(Float)
    DESTINATION_IP = Column(Float)
    DESTINATION_PORT = Column(Float)
    PROTOCOL = Column(Float)
    TIMESTAMP = Column(Float)
    FLOW_DURATION = Column(Float)
    TOTAL_FWD_PACKETS = Column(Float)
    TOTAL_BACKWARD_PACKETS = Column(Float)
    TOTAL_LENGTH_OF_FWD_PACKETS = Column(Float)
    TOTAL_LENGTH_OF_BWD_PACKETS = Column(Float)
    FWD_PACKET_LENGTH_MAX = Column(Float)
    FWD_PACKET_LENGTH_MIN = Column(Float)
    FWD_PACKET_LENGTH_MEAN = Column(Float)
    FWD_PACKET_LENGTH_STD = Column(Float)
    BWD_PACKET_LENGTH_MAX = Column(Float)
    BWD_PACKET_LENGTH_MIN = Column(Float)
    BWD_PACKET_LENGTH_MEAN = Column(Float)
    BWD_PACKET_LENGTH_STD = Column(Float)
    FLOW_BYTES_S = Column(Float)
    FLOW_PACKETS_S = Column(Float)
    FLOW_IAT_MEAN = Column(Float)
    FLOW_IAT_STD = Column(Float)
    FLOW_IAT_MAX = Column(Float)
    FLOW_IAT_MIN = Column(Float)
    FWD_IAT_TOTAL = Column(Float)
    FWD_IAT_MEAN = Column(Float)
    FWD_IAT_STD = Column(Float)
    FWD_IAT_MAX = Column(Float)
    FWD_IAT_MIN = Column(Float)
    BWD_IAT_TOTAL = Column(Float)
    BWD_IAT_MEAN = Column(Float)
    BWD_IAT_STD = Column(Float)
    BWD_IAT_MAX = Column(Float)
    BWD_IAT_MIN = Column(Float)
    FWD_PSH_FLAGS = Column(Float)
    BWD_PSH_FLAGS = Column(Float)
    FWD_URG_FLAGS = Column(Float)
    BWD_URG_FLAGS = Column(Float)
    FWD_HEADER_LENGTH = Column(Float)
    BWD_HEADER_LENGTH = Column(Float)
    FWD_PACKETS_S = Column(Float)
    BWD_PACKETS_S = Column(Float)
    MIN_PACKET_LENGTH = Column(Float)
    MAX_PACKET_LENGTH = Column(Float)
    PACKET_LENGTH_MEAN = Column(Float)
    PACKET_LENGTH_STD = Column(Float)
    PACKET_LENGTH_VARIANCE = Column(Float)
    FIN_FLAG_COUNT = Column(Float)
    SYN_FLAG_COUNT = Column(Float)
    RST_FLAG_COUNT = Column(Float)
    PSH_FLAG_COUNT = Column(Float)
    ACK_FLAG_COUNT = Column(Float)
    URG_FLAG_COUNT = Column(Float)
    CWE_FLAG_COUNT = Column(Float)
    ECE_FLAG_COUNT = Column(Float)
    DOWN_UP_RATIO = Column(Float)
    AVERAGE_PACKET_SIZE = Column(Float)
    AVG_FWD_SEGMENT_SIZE = Column(Float)
    AVG_BWD_SEGMENT_SIZE = Column(Float)
    FWD_AVG_BYTES_BULK = Column(Float)
    FWD_AVG_PACKETS_BULK = Column(Float)
    FWD_AVG_BULK_RATE = Column(Float)
    BWD_AVG_BYTES_BULK = Column(Float)
    BWD_AVG_PACKETS_BULK = Column(Float)
    BWD_AVG_BULK_RATE = Column(Float)
    SUBFLOW_FWD_PACKETS = Column(Float)
    SUBFLOW_FWD_BYTES = Column(Float)
    SUBFLOW_BWD_PACKETS = Column(Float)
    SUBFLOW_BWD_BYTES = Column(Float)
    INIT_WIN_BYTES_FORWARD = Column(Float)
    INIT_WIN_BYTES_BACKWARD = Column(Float)
    ACT_DATA_PKT_FWD = Column(Float)
    MIN_SEG_SIZE_FORWARD = Column(Float)
    ACTIVE_MEAN = Column(Float)
    ACTIVE_STD = Column(Float)
    ACTIVE_MAX = Column(Float)
    ACTIVE_MIN = Column(Float)
    IDLE_MEAN = Column(Float)
    IDLE_STD = Column(Float)
    IDLE_MAX = Column(Float)
    IDLE_MIN = Column(Float)
    LABEL = Column(String)

'''
- 검정된 1차 원천 데이터가 Data-PipeLine 을 통해서 입력되었다고 보면 좋겠지만
그렇지 않은 경우도 많기 때문에 정제되어 있지 않고 모델에 맏게 preprocessing 되어 있지 않게
DB에 저장되어 있다고 가정하자. 

-기본 적인 컬럼명 설정은 root 에 data_explore 사전 탐지한다고 가정
'''


class CSCIDS2017_FRI_PM_PORTSCAN(Base):
    __tablename__ = "CSCIDS2017_FRI_PM_PORTSCAN"
    
    Index                       = Column(Integer, primary_key=True)
    DESTINATION_PORT            = Column(Integer)
    FLOW_DURATION               = Column(Integer)
    TOTAL_FWD_PACKETS           = Column(Integer)
    TOTAL_BACKWARD_PACKETS      = Column(Integer)
    TOTAL_LENGTH_OF_FWD_PACKETS = Column(Integer)
    TOTAL_LENGTH_OF_BWD_PACKETS = Column(Integer)
    FWD_PACKET_LENGTH_MAX       = Column(Integer)
    FWD_PACKET_LENGTH_MIN       = Column(Integer)
    FWD_PACKET_LENGTH_MEAN      = Column(Float)
    FWD_PACKET_LENGTH_STD       = Column(Float)
    BWD_PACKET_LENGTH_MAX       = Column(Integer)
    BWD_PACKET_LENGTH_MIN       = Column(Integer)
    BWD_PACKET_LENGTH_MEAN      = Column(Float)
    BWD_PACKET_LENGTH_STD       = Column(Float)
    FLOW_BYTES_S                = Column(Float)
    FLOW_PACKETS_S              = Column(Float)
    FLOW_IAT_MEAN               = Column(Float)
    FLOW_IAT_STD                = Column(Float)
    FLOW_IAT_MAX                = Column(Integer)
    FLOW_IAT_MIN                = Column(Integer)
    FWD_IAT_TOTAL               = Column(Integer)
    FWD_IAT_MEAN                = Column(Float)
    FWD_IAT_STD                 = Column(Float)
    FWD_IAT_MAX                 = Column(Integer)
    FWD_IAT_MIN                 = Column(Integer)
    BWD_IAT_TOTAL               = Column(Integer)
    BWD_IAT_MEAN                = Column(Float)
    BWD_IAT_STD                 = Column(Float)
    BWD_IAT_MAX                 = Column(Integer)
    BWD_IAT_MIN                 = Column(Integer)
    FWD_PSH_FLAGS               = Column(Integer)
    BWD_PSH_FLAGS               = Column(Integer)
    FWD_URG_FLAGS               = Column(Integer)
    BWD_URG_FLAGS               = Column(Integer)
    FWD_HEADER_LENGTH           = Column(Integer)
    BWD_HEADER_LENGTH           = Column(Integer)
    FWD_PACKETS_S               = Column(Float)
    BWD_PACKETS_S               = Column(Float)
    MIN_PACKET_LENGTH           = Column(Integer)
    MAX_PACKET_LENGTH           = Column(Integer)
    PACKET_LENGTH_MEAN          = Column(Float)
    PACKET_LENGTH_STD           = Column(Float)
    PACKET_LENGTH_VARIANCE      = Column(Float)
    FIN_FLAG_COUNT              = Column(Integer)
    SYN_FLAG_COUNT              = Column(Integer)
    RST_FLAG_COUNT              = Column(Integer)
    PSH_FLAG_COUNT              = Column(Integer)
    ACK_FLAG_COUNT              = Column(Integer)
    URG_FLAG_COUNT              = Column(Integer)
    CWE_FLAG_COUNT              = Column(Integer)
    ECE_FLAG_COUNT              = Column(Integer)
    DOWN_UP_RATIO               = Column(Integer)
    AVERAGE_PACKET_SIZE         = Column(Float)
    AVG_FWD_SEGMENT_SIZE        = Column(Float)
    AVG_BWD_SEGMENT_SIZE        = Column(Float)
    FWD_HEADER_LENGTH_1         = Column(Integer)
    FWD_AVG_BYTES_BULK          = Column(Integer)
    FWD_AVG_PACKETS_BULK        = Column(Integer)
    FWD_AVG_BULK_RATE           = Column(Integer)
    BWD_AVG_BYTES_BULK          = Column(Integer)
    BWD_AVG_PACKETS_BULK        = Column(Integer)
    BWD_AVG_BULK_RATE           = Column(Integer)
    SUBFLOW_FWD_PACKETS         = Column(Integer)
    SUBFLOW_FWD_BYTES           = Column(Integer)
    SUBFLOW_BWD_PACKETS         = Column(Integer)
    SUBFLOW_BWD_BYTES           = Column(Integer)
    INIT_WIN_BYTES_FORWARD      = Column(Integer)
    INIT_WIN_BYTES_BACKWARD     = Column(Integer)
    ACT_DATA_PKT_FWD            = Column(Integer)
    MIN_SEG_SIZE_FORWARD        = Column(Integer)
    ACTIVE_MEAN                 = Column(Float)
    ACTIVE_STD                  = Column(Float)
    ACTIVE_MAX                  = Column(Integer)
    ACTIVE_MIN                  = Column(Integer)
    IDLE_MEAN                   = Column(Float)
    IDLE_STD                    = Column(Float)
    IDLE_MAX                    = Column(Integer)
    IDLE_MIN                    = Column(Integer)
    LABEL                       = Column(String)



    
# class CSCIDS2017_fri_pm(Base):
#     __tablename__ = "CSCIDS2017_fri_pm"

# class CSCIDS2017_fri_pm(Base):
#     __tablename__ = "CSCIDS2017_fri_pm"       

# class CSCIDS2017_fri_pm(Base):
#     __tablename__ = "CSCIDS2017_fri_pm"

# class CSCIDS2017_fri_pm(Base):
#     __tablename__ = "CSCIDS2017_fri_pm"

# class CSCIDS2017_fri_pm(Base):
#     __tablename__ = "CSCIDS2017_fri_pm"

# class CSCIDS2017_fri_pm(Base):
#     __tablename__ = "CSCIDS2017_fri_pm"

# class CSCIDS2017_fri_pm(Base):
#     __tablename__ = "CSCIDS2017_fri_pm"    