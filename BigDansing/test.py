a = [[1,2],[3,4]]
for aa in a:
    try:
        aa.remove(1)
    except Exception as e:
        continue
print(a)