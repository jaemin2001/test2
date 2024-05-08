import uuid
from pydantic import BaseModel, Field

""" DB model connect sqlalchemy """

class Base_model(BaseModel):
    """Base_model"""
    pk: uuid.UUID
    name: str
    
    class Config:
        from_attributes : True


""" model Input Data specify """

class DataInput(BaseModel):
    NM : str = Field(min_length=4, max_length=10)
    x : list[float] = Field()

class PredictOutput(BaseModel):
    prediction : int

""" 
data name : CSCIDS2017
target file count : 8 -> table 8s
columns count  : 79 * 8  : 632
total row count : ??    

sample :> balance data file : 1

"""
class CSCIDS_balanced(BaseModel):
    """CSCIDS2017_BALANCED_ATTK"""
    Index : int      
    FLOW_ID : float
    SOURCE_IP : float
    SOURCE_PORT : float
    DESTINATION_IP : float
    DESTINATION_PORT : float
    PROTOCOL : float
    TIMESTAMP : float
    FLOW_DURATION : float
    TOTAL_FWD_PACKETS : float
    TOTAL_BACKWARD_PACKETS : float
    TOTAL_LENGTH_OF_FWD_PACKETS : float
    TOTAL_LENGTH_OF_BWD_PACKETS : float
    FWD_PACKET_LENGTH_MAX : float
    FWD_PACKET_LENGTH_MIN : float
    FWD_PACKET_LENGTH_MEAN : float
    FWD_PACKET_LENGTH_STD : float
    BWD_PACKET_LENGTH_MAX : float
    BWD_PACKET_LENGTH_MIN : float
    BWD_PACKET_LENGTH_MEAN : float
    BWD_PACKET_LENGTH_STD : float
    FLOW_BYTES_S : float
    FLOW_PACKETS_S : float
    FLOW_IAT_MEAN : float
    FLOW_IAT_STD : float
    FLOW_IAT_MAX : float
    FLOW_IAT_MIN : float
    FWD_IAT_TOTAL : float
    FWD_IAT_MEAN : float
    FWD_IAT_STD : float
    FWD_IAT_MAX : float
    FWD_IAT_MIN : float
    BWD_IAT_TOTAL : float
    BWD_IAT_MEAN : float
    BWD_IAT_STD : float
    BWD_IAT_MAX : float
    BWD_IAT_MIN : float
    FWD_PSH_FLAGS : float
    BWD_PSH_FLAGS : float
    FWD_URG_FLAGS : float
    BWD_URG_FLAGS : float
    FWD_HEADER_LENGTH : float
    BWD_HEADER_LENGTH : float
    FWD_PACKETS_S : float
    BWD_PACKETS_S : float
    MIN_PACKET_LENGTH : float
    MAX_PACKET_LENGTH : float
    PACKET_LENGTH_MEAN : float
    PACKET_LENGTH_STD : float
    PACKET_LENGTH_VARIANCE : float
    FIN_FLAG_COUNT : float
    SYN_FLAG_COUNT : float
    RST_FLAG_COUNT : float
    PSH_FLAG_COUNT : float
    ACK_FLAG_COUNT : float
    URG_FLAG_COUNT : float
    CWE_FLAG_COUNT : float
    ECE_FLAG_COUNT : float
    DOWN_UP_RATIO : float
    AVERAGE_PACKET_SIZE : float
    AVG_FWD_SEGMENT_SIZE : float
    AVG_BWD_SEGMENT_SIZE : float
    FWD_AVG_BYTES_BULK : float
    FWD_AVG_PACKETS_BULK : float
    FWD_AVG_BULK_RATE : float
    BWD_AVG_BYTES_BULK : float
    BWD_AVG_PACKETS_BULK : float
    BWD_AVG_BULK_RATE : float
    SUBFLOW_FWD_PACKETS : float
    SUBFLOW_FWD_BYTES : float
    SUBFLOW_BWD_PACKETS : float
    SUBFLOW_BWD_BYTES : float
    INIT_WIN_BYTES_FORWARD : float
    INIT_WIN_BYTES_BACKWARD : float
    ACT_DATA_PKT_FWD : float
    MIN_SEG_SIZE_FORWARD : float
    ACTIVE_MEAN : float
    ACTIVE_STD : float
    ACTIVE_MAX : float
    ACTIVE_MIN : float
    IDLE_MEAN : float
    IDLE_STD : float
    IDLE_MAX : float
    IDLE_MIN : float
    LABE : str
    
    class Config:
        from_attributes : True

        
class CSCIDS2017_fri_pm_postscan(BaseModel):
    """CSCIDS2017_FRI_PM_PORTSCAN"""
    Index                       :int
    DESTINATION_PORT            :int
    FLOW_DURATION               :int
    TOTAL_FWD_PACKETS           :int
    TOTAL_BACKWARD_PACKETS      :int
    TOTAL_LENGTH_OF_FWD_PACKETS :int
    TOTAL_LENGTH_OF_BWD_PACKETS :int
    FWD_PACKET_LENGTH_MAX       :int
    FWD_PACKET_LENGTH_MIN       :int
    FWD_PACKET_LENGTH_MEAN      :float
    FWD_PACKET_LENGTH_STD       :float
    BWD_PACKET_LENGTH_MAX       :int
    BWD_PACKET_LENGTH_MIN       :int
    BWD_PACKET_LENGTH_MEAN      :float
    BWD_PACKET_LENGTH_STD       :float
    FLOW_BYTES_S                :float
    FLOW_PACKETS_S              :float
    FLOW_IAT_MEAN               :float
    FLOW_IAT_STD                :float
    FLOW_IAT_MAX                :int
    FLOW_IAT_MIN                :int
    FWD_IAT_TOTAL               :int
    FWD_IAT_MEAN                :float
    FWD_IAT_STD                 :float
    FWD_IAT_MAX                 :int
    FWD_IAT_MIN                 :int
    BWD_IAT_TOTAL               :int
    BWD_IAT_MEAN                :float
    BWD_IAT_STD                 :float
    BWD_IAT_MAX                 :int
    BWD_IAT_MIN                 :int
    FWD_PSH_FLAGS               :int
    BWD_PSH_FLAGS               :int
    FWD_URG_FLAGS               :int
    BWD_URG_FLAGS               :int
    FWD_HEADER_LENGTH           :int
    BWD_HEADER_LENGTH           :int
    FWD_PACKETS_S               :float
    BWD_PACKETS_S               :float
    MIN_PACKET_LENGTH           :int
    MAX_PACKET_LENGTH           :int
    PACKET_LENGTH_MEAN          :float
    PACKET_LENGTH_STD           :float
    PACKET_LENGTH_VARIANCE      :float
    FIN_FLAG_COUNT              :int
    SYN_FLAG_COUNT              :int
    RST_FLAG_COUNT              :int
    PSH_FLAG_COUNT              :int
    ACK_FLAG_COUNT              :int
    URG_FLAG_COUNT              :int
    CWE_FLAG_COUNT              :int
    ECE_FLAG_COUNT              :int
    DOWN_UP_RATIO               :int
    AVERAGE_PACKET_SIZE         :float
    AVG_FWD_SEGMENT_SIZE        :float
    AVG_BWD_SEGMENT_SIZE        :float
    FWD_HEADER_LENGTH_1         :int
    FWD_AVG_BYTES_BULK          :int
    FWD_AVG_PACKETS_BULK        :int
    FWD_AVG_BULK_RATE           :int
    BWD_AVG_BYTES_BULK          :int
    BWD_AVG_PACKETS_BULK        :int
    BWD_AVG_BULK_RATE           :int
    SUBFLOW_FWD_PACKETS         :int
    SUBFLOW_FWD_BYTES           :int
    SUBFLOW_BWD_PACKETS         :int
    SUBFLOW_BWD_BYTES           :int
    INIT_WIN_BYTES_FORWARD      :int
    INIT_WIN_BYTES_BACKWARD     :int
    ACT_DATA_PKT_FWD            :int
    MIN_SEG_SIZE_FORWARD        :int
    ACTIVE_MEAN                 :float
    ACTIVE_STD                  :float
    ACTIVE_MAX                  :int
    ACTIVE_MIN                  :int
    IDLE_MEAN                   :float
    IDLE_STD                    :float
    IDLE_MAX                    :int
    IDLE_MIN                    :int
    LABEL                       :str
    
    class Config:
        from_attributes : True
