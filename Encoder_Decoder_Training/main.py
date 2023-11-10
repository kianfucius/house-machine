from Encoder_Decoder_Training.Data_Processing import pre_processor
from torch.utils.data import DataLoader

if __name__ == '__main__':
    preprocessor = pre_processor.PreProcessor()
    preprocessor.preprocess(verbose= True)
    train_set , val_set = preprocessor.split_into_train_val(0.5)
    #train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(val_set, batch_size=64, shuffle=True)
    tesnsor, dict_var = next(iter(test_dataloader))
    print('program finished')