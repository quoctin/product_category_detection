
import re
def get_data_lst(file_):
    # read file paths
    f = open(file_, 'r')
    file_lst = []
    line = f.readline().strip('\n ')
    if line is not '':
        file_lst.append(line)
    while line:
        line = f.readline().strip('\n ')
        if line is not '':
            file_lst.append(line)
    f.close()
    return file_lst

def glob_re(pattern, strings):
    return list(filter(re.compile(pattern).match, strings))