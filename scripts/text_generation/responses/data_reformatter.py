import json
from pprint import pprint
file = "lore.json"

file_content = open(file, 'r')
file_content = json.load(file_content)


# for responses in file_content["responses"]:
#     for responses_l2 in responses["responses"]:
#         #print(responses_l2)
#         for i in range(len(responses_l2)):
#
#             print(responses_l2[i])
#
#             content = {
#                 "text": responses_l2[i],
#                 "expression": None
#             }
#
#             responses_l2[i] = content
#
# json_object = json.dumps(file_content, indent=2)
# print(json_object)
#
# with open(file, "w") as f:
#     f.write(json_object)
