from Data_Processing import pre_processor
from torch.utils.data import DataLoader
import datetime

if __name__ == "__main__":
    preprocessor = pre_processor.PreProcessor()
    preprocessor.preprocess(verbose=True)
    train_set, val_set = preprocessor.split_into_train_val(0.5)
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataloader = DataLoader(
        val_set,
        batch_size=128,
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
    )
    total = 0

    print(next(iter(test_dataloader)).shape)
    for i in range(0, 20):
        start = datetime.datetime.now()
        next(iter(test_dataloader))
        total = total + (datetime.datetime.now() - start).total_seconds()

    print(f"Average: {total/20}")
