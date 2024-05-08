import pandas as pd


df = pd.read_csv('./DW/storage/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
print(df.info())

'''

 #   Column                        Non-Null Count   Dtype  
---  ------                        --------------   -----  
 0    Destination Port             286467 non-null  int64  
 1    Flow Duration                286467 non-null  int64  
 2    Total Fwd Packets            286467 non-null  int64  
 3    Total Backward Packets       286467 non-null  int64  
 4   Total Length of Fwd Packets   286467 non-null  int64  
 5    Total Length of Bwd Packets  286467 non-null  int64  
 6    Fwd Packet Length Max        286467 non-null  int64  
 7    Fwd Packet Length Min        286467 non-null  int64  
 8    Fwd Packet Length Mean       286467 non-null  float64
 9    Fwd Packet Length Std        286467 non-null  float64
 10  Bwd Packet Length Max         286467 non-null  int64  
 11   Bwd Packet Length Min        286467 non-null  int64  
 12   Bwd Packet Length Mean       286467 non-null  float64
 13   Bwd Packet Length Std        286467 non-null  float64
 14  Flow Bytes/s                  286452 non-null  float64
 15   Flow Packets/s               286467 non-null  float64
 16   Flow IAT Mean                286467 non-null  float64
 17   Flow IAT Std                 286467 non-null  float64
 18   Flow IAT Max                 286467 non-null  int64
 19   Flow IAT Min                 286467 non-null  int64
 20  Fwd IAT Total                 286467 non-null  int64
 21   Fwd IAT Mean                 286467 non-null  float64
 22   Fwd IAT Std                  286467 non-null  float64
 23   Fwd IAT Max                  286467 non-null  int64
 24   Fwd IAT Min                  286467 non-null  int64
 25  Bwd IAT Total                 286467 non-null  int64
 26   Bwd IAT Mean                 286467 non-null  float64
 27   Bwd IAT Std                  286467 non-null  float64
 28   Bwd IAT Max                  286467 non-null  int64
 29   Bwd IAT Min                  286467 non-null  int64
 30  Fwd PSH Flags                 286467 non-null  int64
 31   Bwd PSH Flags                286467 non-null  int64
 32   Fwd URG Flags                286467 non-null  int64
 33   Bwd URG Flags                286467 non-null  int64
 34   Fwd Header Length            286467 non-null  int64
 35   Bwd Header Length            286467 non-null  int64
 36  Fwd Packets/s                 286467 non-null  float64
 37   Bwd Packets/s                286467 non-null  float64
 38   Min Packet Length            286467 non-null  int64
 39   Max Packet Length            286467 non-null  int64
 40   Packet Length Mean           286467 non-null  float64
 41   Packet Length Std            286467 non-null  float64
 42   Packet Length Variance       286467 non-null  float64
 43  FIN Flag Count                286467 non-null  int64
 44   SYN Flag Count               286467 non-null  int64
 45   RST Flag Count               286467 non-null  int64
 46   PSH Flag Count               286467 non-null  int64
 47   ACK Flag Count               286467 non-null  int64
 48   URG Flag Count               286467 non-null  int64
 49   CWE Flag Count               286467 non-null  int64
 50   ECE Flag Count               286467 non-null  int64
 51   Down/Up Ratio                286467 non-null  int64
 52   Average Packet Size          286467 non-null  float64
 53   Avg Fwd Segment Size         286467 non-null  float64
 54   Avg Bwd Segment Size         286467 non-null  float64
 55   Fwd Header Length.1          286467 non-null  int64
 56  Fwd Avg Bytes/Bulk            286467 non-null  int64
 57   Fwd Avg Packets/Bulk         286467 non-null  int64
 58   Fwd Avg Bulk Rate            286467 non-null  int64
 59   Bwd Avg Bytes/Bulk           286467 non-null  int64
 60   Bwd Avg Packets/Bulk         286467 non-null  int64
 61  Bwd Avg Bulk Rate             286467 non-null  int64
 62  Subflow Fwd Packets           286467 non-null  int64
 63   Subflow Fwd Bytes            286467 non-null  int64
 64   Subflow Bwd Packets          286467 non-null  int64
 65   Subflow Bwd Bytes            286467 non-null  int64
 66  Init_Win_bytes_forward        286467 non-null  int64
 67   Init_Win_bytes_backward      286467 non-null  int64
 68   act_data_pkt_fwd             286467 non-null  int64
 69   min_seg_size_forward         286467 non-null  int64
 70  Active Mean                   286467 non-null  float64
 71   Active Std                   286467 non-null  float64
 72   Active Max                   286467 non-null  int64
 73   Active Min                   286467 non-null  int64
 74  Idle Mean                     286467 non-null  float64
 75   Idle Std                     286467 non-null  float64
 76   Idle Max                     286467 non-null  int64
 77   Idle Min                     286467 non-null  int64
 78   Label                        286467 non-null  object
 
 '''

 
# df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# for i in df.columns:
#     print(i)

'''
DESTINATION_PORT    
FLOW_DURATION   
TOTAL_FWD_PACKETS   
TOTAL_BACKWARD_PACKETS  
TOTAL_LENGTH_OF_FWD_PACKETS 
TOTAL_LENGTH_OF_BWD_PACKETS 
FWD_PACKET_LENGTH_MAX   
FWD_PACKET_LENGTH_MIN   
FWD_PACKET_LENGTH_MEAN  
FWD_PACKET_LENGTH_STD   
BWD_PACKET_LENGTH_MAX   
BWD_PACKET_LENGTH_MIN   
BWD_PACKET_LENGTH_MEAN  
BWD_PACKET_LENGTH_STD   
FLOW_BYTES/S    
FLOW_PACKETS/S  
FLOW_IAT_MEAN   
FLOW_IAT_STD    
FLOW_IAT_MAX    
FLOW_IAT_MIN    
FWD_IAT_TOTAL   
FWD_IAT_MEAN    
FWD_IAT_STD 
FWD_IAT_MAX 
FWD_IAT_MIN 
BWD_IAT_TOTAL   
BWD_IAT_MEAN    
BWD_IAT_STD 
BWD_IAT_MAX 
BWD_IAT_MIN 
FWD_PSH_FLAGS   
BWD_PSH_FLAGS   
FWD_URG_FLAGS   
BWD_URG_FLAGS   
FWD_HEADER_LENGTH   
BWD_HEADER_LENGTH   
FWD_PACKETS/S   
BWD_PACKETS/S   
MIN_PACKET_LENGTH   
MAX_PACKET_LENGTH   
PACKET_LENGTH_MEAN  
PACKET_LENGTH_STD   
PACKET_LENGTH_VARIANCE  
FIN_FLAG_COUNT  
SYN_FLAG_COUNT  
RST_FLAG_COUNT  
PSH_FLAG_COUNT  
ACK_FLAG_COUNT  
URG_FLAG_COUNT  
CWE_FLAG_COUNT  
ECE_FLAG_COUNT  
DOWN/UP_RATIO   
AVERAGE_PACKET_SIZE 
AVG_FWD_SEGMENT_SIZE    
AVG_BWD_SEGMENT_SIZE    
FWD_HEADER_LENGTH.1 
FWD_AVG_BYTES/BULK  
FWD_AVG_PACKETS/BULK    
FWD_AVG_BULK_RATE   
BWD_AVG_BYTES/BULK  
BWD_AVG_PACKETS/BULK    
BWD_AVG_BULK_RATE   
SUBFLOW_FWD_PACKETS 
SUBFLOW_FWD_BYTES   
SUBFLOW_BWD_PACKETS 
SUBFLOW_BWD_BYTES   
INIT_WIN_BYTES_FORWARD  
INIT_WIN_BYTES_BACKWARD 
ACT_DATA_PKT_FWD    
MIN_SEG_SIZE_FORWARD    
ACTIVE_MEAN 
ACTIVE_STD  
ACTIVE_MAX  
ACTIVE_MIN  
IDLE_MEAN   
IDLE_STD    
IDLE_MAX    
IDLE_MIN    
LABEL   
''' 

