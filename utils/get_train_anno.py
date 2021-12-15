import os
import pandas as pd
    

def train_anno_csv():   
    tags = []
    imgnames = []

    for imgname in os.listdir("./data/cats-vs-dogs/train"):
        tag = 0 if imgname.split('.')[0] == "dog" else 1
        imgnames.append(imgname)
        tags.append(tag)

    dic = {'Image': imgnames, 'Tag': tags}

    # Genarate annotation CSV file
    df = pd.DataFrame(dic)
    df.to_csv('./data/cats-vs-dogs/train_anno.csv', index=False)
    return df
