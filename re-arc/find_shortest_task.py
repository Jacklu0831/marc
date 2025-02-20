import glob

json_file_and_count = []
for json_file in glob.glob('train_data/tasks/*'):
    x = open(json_file, 'r').readlines()
    assert len(x) == 1
    json_file_and_count.append((len(x[0]), json_file))

for c, f in sorted(json_file_and_count)[:10]:
    print(c, f)
