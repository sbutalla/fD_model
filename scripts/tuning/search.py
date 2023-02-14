import os
root_dir = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/aggregate_tuning'


def main():
    file_list = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if 'model' in file:
                # print(os.path.join(subdir, file))
                file_list.append(os.path.join(subdir, file))
                
    for file in file_list:
        print(file)
    total = 6*6*5*5*3
    print(len(file_list))
    print(total)
    print(len(file_list)/total)
	
    for file in file_list:
        with open(root_dir) as json_file:
            path_dict =json.load(json_file)


main()
