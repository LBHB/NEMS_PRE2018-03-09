db = {}
with open ("database_info.txt","r") as f:
    for line in f:
        key,val = line.split()
        db[key] = val