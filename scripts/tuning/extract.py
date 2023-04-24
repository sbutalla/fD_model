# import json
# import pprint
# import os
#
# parent_file = '/Users/spencerhirsch/Documents/research/output.json'
#
#
# def main():
#     with open(parent_file) as json_file:
#         list = json.load(json_file)
#     l1 = 5
#     l2 = 5
#     isolated_list = []
#     for item in list:
#         if item['l1'] == l1 and item['l2'] == l2:
#             isolated_list.append(item)
#
#     pprint.pprint(isolated_list)
#     root_dir = '/Users/spencerhirsch/Documents/research/isolated_jsons/'
#     class_out = root_dir + ('%s_%s_models.json' % (l1, l2))
#     out_file = open(class_out, "w")
#     json.dump(isolated_list, out_file, indent=4)
#
#
# main()


import json
import pprint
import os

parent_file = '/Users/spencerhirsch/Documents/research/output.json'


def main():
    with open(parent_file) as json_file:
        list = json.load(json_file)

    optimal = {}
    current = 1
    for item in list:
        if item['eta'] == 0.3 and item['max depth'] == 6 and item['l1'] == 0 and item['l2'] == 1:
            optimal = item

    pprint.pprint(optimal)

main()
