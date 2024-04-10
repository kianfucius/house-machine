from Data_Processing_DMAE.pre_processor import PreProcessor
from torch.utils.data import DataLoader

if __name__ == "__main__":
    BATCH_SIZE = 128
    train_split_prop = 0.999
    preprocessor = PreProcessor()

    train_set, val_set = preprocessor.construct_train_split_data_files(
        train_prop=train_split_prop
    )

    # train_set, val_set = preprocessor.split_into_train_val(train_prop=train_split_prop)

    train_dataloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=5,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=3,
        persistent_workers=True,
    )
    print("finished")
