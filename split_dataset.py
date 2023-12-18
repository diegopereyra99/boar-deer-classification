import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

imgs_path = "data/imgs"
os.listdir(imgs_path)

df = pd.DataFrame(os.listdir(imgs_path), columns=['filename'])
df["class"] = df["filename"].str.split("_").str[0]

df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=0)

df_train.to_csv("data/train.csv", index=False, header=False)
df_val.to_csv("data/val.csv", index=False, header=False)


# os.makedirs("data/dataset", exist_ok=True)
# for st, df_st in zip(["train", "val"], [df_train, df_val]):
#     os.makedirs(f"data/dataset/{st}", exist_ok=True)
#     for cl, dff in df_st.groupby("class"):
#         dst_path = f"data/dataset/{st}/{cl}"
#         os.makedirs(dst_path, exist_ok=True)
#         for img_fn in dff["filename"]:
#             shutil.copy(
#                 os.path.join(imgs_path, img_fn),
#                 os.path.join(dst_path)
#             )
            
            