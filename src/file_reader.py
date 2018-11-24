class FileReader():

    def __init__(self):
        pass

    def read(self, file_path):
        file_content = open(file_path)
        file_list = file_content.readlines()
        file_content.close
        return file_list
