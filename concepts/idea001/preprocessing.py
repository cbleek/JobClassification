import pandas as pd
import json
import pandas
import codecs
import re
import glob
from sklearn.utils import shuffle


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    text = text.replace("\n", "").strip()
    return re.sub(clean, '', text)


requirements = []
employmentType = []

for filepath in glob.iglob('data/*.json'):
    file = codecs.open(filepath, "r")
    jobs = json.load(file)
    for job in jobs["jobs"]:
        if "classifications" in job and job["templateValues"]["requirements"]:
            employmentType.append(job['classifications']['employmentTypes'])
            requirements.append(remove_html_tags(job["templateValues"]["requirements"]))

df = pandas.DataFrame({"employmentType": employmentType, "description": requirements})
df.to_csv("data/jobs.csv", index=False)


files = glob.glob("data/*.csv")

list_ = []

for file_ in files:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)

frame = pd.concat(list_, axis=0, ignore_index=True)
frame = shuffle(frame)
frame.to_csv("data/data.csv", index=False)
