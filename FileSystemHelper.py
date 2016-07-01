import sys
import os

class FileSystemHelper:
    def __init__(self, directory, file_ext):
        self.directory = directory
        self.file_ext = file_ext
        self.all_files = []
        for file_ in os.listdir(self.directory):
            if file_.endswith(self.file_ext):
                self.all_files.append(file_)

    def for_each_file_execute_this(self, callback):
        for file_ in self.all_files:
            file_path = self.directory + "/" + file_
            callback(file_path, file_)
