import os
def create_file_or_folder(location, name, is_folder=False):
    if is_folder:
        path = os.path.join(location, name)
        os.makedirs(path, exist_ok=True)
    else:
        path = os.path.join(location, name)
        with open(path, 'w'):
            pass
