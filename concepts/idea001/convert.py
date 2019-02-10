import json
import pandas
import codecs
import re
import glob


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    text = text.replace("\n", "").strip()
    return re.sub(clean, '', text)


requirements = []
titles = []

for filepath in glob.iglob('data/*.json'):
    file = codecs.open(filepath, "r")
    jobs = json.load(file)
    for job in jobs["jobs"]:
        if "classifications" in job and job["templateValues"]["requirements"]:
            titles.append(job['classifications']['employmentTypes'])
            requirements.append(remove_html_tags(job["templateValues"]["requirements"]))

df = pandas.DataFrame({"title": titles, "requirement": requirements})
df.to_csv("data/jobs.csv", index=False)



