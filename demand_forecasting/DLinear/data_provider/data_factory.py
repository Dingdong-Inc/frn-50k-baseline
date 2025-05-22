from data_provider.data_loader import Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = Dataset_Custom
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        data_path=args.data_path,
        flag=flag,
        total_seq_len=args.total_seq_len,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        train_only=train_only,
        scale=args.scale
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
