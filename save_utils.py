import os
import pickle

ROOT_PATH = "./save/"
RECENT_SUFFIX = "recent/"
ARCHIVE_SUFFIX = "archive/"
VERSION_FILE_NAME = "version.txt"


def get_full_path(data_name):
    return ROOT_PATH + data_name + "/"


def get_archive_path(data_name):
    return ROOT_PATH + data_name + "/" + ARCHIVE_SUFFIX


def get_recent_path(data_name):
    return ROOT_PATH + data_name + "/" + RECENT_SUFFIX


def get_version_file_path(data_name):
    return ROOT_PATH + data_name + "/" + VERSION_FILE_NAME


def get_data_file_name(data_name, k):
    return data_name + "-" + str(k) + ".pkl"


def create_new_data_path(data_name):
    full_path = get_full_path(data_name)
    os.makedirs(full_path)
    version_num_file_path = full_path + VERSION_FILE_NAME
    with open(version_num_file_path, "w") as f:
        f.write("0")
    archive_path = full_path+ARCHIVE_SUFFIX
    recent_path = full_path+RECENT_SUFFIX
    os.makedirs(archive_path)
    os.makedirs(recent_path)


def get_version_number(data_name):
    version_num_file_path = get_version_file_path(data_name)
    with open(version_num_file_path, "r") as f:
        current_file_num = int(f.read())
    return current_file_num


def increment_version_number(data_name):
    version_num_file_path = get_version_file_path(data_name)
    current_file_num = get_version_number(data_name)
    new_file_num = current_file_num + 1
    with open(version_num_file_path, "w") as f:
        f.write(str(new_file_num))
    f.close()


def save_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def move_last_run_to_archive(data_name):
    recent_path = get_recent_path(data_name)
    archive_path = get_archive_path(data_name)
    recent_files = os.listdir(recent_path)
    most_recent_file = recent_files[0]
    most_recent_file_path = recent_path+most_recent_file
    archive_file_path = archive_path+most_recent_file
    os.rename(most_recent_file_path, archive_file_path)


def save_data(data_name, data):
    """
    Saves a piece of data in a pickle file in the ./save directory. The data_name should be a
    unique identifier for the piece of data to be saved. The file will be saved in ./save/data_name/recent/data_name-x.pkl
    where x is the version number of the file. The most recent save of this data will reside in the recent directory and
    all older saves will be in ./save/data_name/archive. x increments by 1 every time the data is saved. The x is stored in
    ./save/data_name/version.txt. Manually editing this file will cause the function to break.
    :param data_name: unique identifier in the file system for the type of data being archived (e.g. "bleu_scores")
    :param data: the variable data to save in the file
    :return: void
    """
    full_path = ROOT_PATH + data_name + "/"
    recent_path = get_recent_path(data_name)
    if not os.path.exists(full_path):
        create_new_data_path(data_name)
        data_path = recent_path + data_name + "-0.pkl"
        save_file(data_path, data)
    else:
        move_last_run_to_archive(data_name)
        current_version_num = get_version_number(data_name)
        data_path = recent_path + get_data_file_name(data_name, current_version_num)
    save_file(data_path, data)
    increment_version_number(data_name)


def load_version_number(data_name, version_number):
    """
    Loads the file with version number version_number
    :param data_name: the uq identifier for the data that is to be loaded
    :param version_number: the version of the data that is to be loaded.
    :return: the data stored in the requested file. Raises exception if invalid broken number.
    """
    current_version_num = get_version_number(data_name)
    if version_number >= current_version_num:
        raise Exception("version doesn't exist")
    file_name = get_data_file_name(data_name, version_number)
    if version_number == current_version_num-1:
        recent_path = get_recent_path(data_name)
        data_path = recent_path + file_name
    else:
        archive_path = get_archive_path(data_name)
        data_path = archive_path + file_name
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_most_recent_file(data_name):
    """
    Loads the most recent file for data_name.
    :param data_name: the uq identifier for the data that is to be loaded
    :return: the data stored in the requested file.
    """
    recent_version_num = get_version_number(data_name) - 1
    return load_version_number(data_name, recent_version_num)


if __name__ == "__main__":
    print(load_most_recent_file("test"))
