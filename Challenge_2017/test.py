from Challenge_2017.data_parser import Data_Parser

train_data = "/home/linhhima/Atrial-Fibrillation-Detection-from-ECG-via-Machine-Learning-Methods/training2017"
electrocardiogram_sample_rate = 300
my_data_parser = Data_Parser(train_data, electrocardiogram_sample_rate)
my_data_parser.read_training_dataset()

test_data = "/home/linhhima/Atrial-Fibrillation-Detection-from-ECG-via-Machine-Learning-Methods/validation"
my_test_parser = Data_Parser(test_data, electrocardiogram_sample_rate)
my_test_parser.read_testing_dataset()
