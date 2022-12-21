id_ip = {
    '0': '10.10.114.14',
    '1': '10.10.114.14',
    '2': '10.10.114.14',
    '3': '10.10.114.14'
}

id_name = {
    '0':'云端',
    '1':'无人机-1',
    '2':'无人机-2',
    '3':'无人机-3',
}
ip_id = {}
for key, val in id_ip.items():
    ip_id[val] = key
name_id = {}
for key, val in id_name.items():
    name_id[val] = key