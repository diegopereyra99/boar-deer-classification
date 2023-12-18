import os
from simple_image_download import simple_image_download as simp 
response = simp.simple_image_download

keys = ["male", "female", "baby", "young", "eating", "animal", "wild", "eating grass", "old", "america", "europe", "far away", "group", "hunt", "africa"]

google = response().urls("boar", 10)[:6]

for a in ["boar", "deer"]:
    if os.path.exists(f"data/sources/{a}_urls.txt"):
        with open(f"data/sources/{a}_urls.txt", "r") as file:
            lst = [fn.strip() for fn in file.readlines()]
    else:
        lst = []
        
    for k in keys:
        lst_a = response().urls(f"{a} {k}", 150)
        lst.extend(lst_a[5:])
        
    lst2 = list(set(lst))
    lst2 = result_list = [item for item in lst2 if item not in google]
    print(len(lst), len(lst2))
    with open(f"data/sources/{a}_urls.txt", "w") as f:
        f.writelines([fn + "\n" for fn in lst2])