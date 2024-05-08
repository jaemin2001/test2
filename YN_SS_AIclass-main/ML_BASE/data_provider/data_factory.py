from data_provider.data_loader import Dataset_BIVA, Dataset_Pred
from data_provider.data_loader_CSCIDS import Dataset_CSCIDS2017, Dataset_CSCIDS2017_Pred
from torch.utils.data import DataLoader

data_dict = {
    'BIVA': [Dataset_BIVA, Dataset_Pred],
    # 'stats_DFM' : Dataset_DFM
    'CSCIDS2017' : [Dataset_CSCIDS2017, Dataset_CSCIDS2017_Pred],
}

def data_provider(args, flag):
    Data = data_dict[args.data][0]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = data_dict[args.data][1]

    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        flag=flag,
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        timeenc=timeenc,
        freq=freq,
        cols=args.cols,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader
