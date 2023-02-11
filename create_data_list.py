from utils.data_utils import split_data_to_train_test, create_enroll_from_json
from models.speechbrain_ecpa import AttackSRModel


# should run only once when using new data to create train and test partition and corespending lists
# create enroll list using specific speaker recognition model

DATA_PATH = "data/LIBRI/d3"
def main():
    '''
    split data to train and test, storing train and test lists in parent path
    :return: json files
    '''
    data_root_path = DATA_PATH
    split_data_to_train_test(data_root_path)

    '''
    create enroll list using attackSR model. should define a model  
    :return: pickle and json file 
    '''
    model_instance = AttackSRModel()
    train_list_path = "data/LIBRI/data_lists/d3/train_files.json"
    create_enroll_from_json(train_list_path, model_instance)


if __name__ == '__main__':
    main()
