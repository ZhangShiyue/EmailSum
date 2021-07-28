import os
import re
import codecs
import numpy as np
import datetime
import ujson as json
from tqdm import tqdm


def unzip_files():
    os.system("tar -xzvf raw_data/w3c-emails-part1.tar.gz")
    os.system("tar -xzvf raw_data/w3c-emails-part2.tar.gz")
    os.system("mv lists-* w3c-emails")
    os.system("tar -xzvf raw_data/w3c-emails-part3.tar.gz")
    os.system("mv lists-* w3c-emails")


def group_emails_by_subjects():
    print("====Group emails by their subjects and save to subject.json====")
    subjects = {}
    files = os.listdir("w3c-emails")
    model1 = re.compile(r'[a-zA-Z0-9]')
    model2 = re.compile(r'[^\x00-\x7f]')
    for file in tqdm(files):
        with codecs.open(f"w3c-emails/{file}", 'r', encoding='utf-8', errors='ignore')as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("subject"):
                    subject = line.split('=')[1].replace('"', '').lower()
                    if '?????' in subject:
                        break
                    subject = subject.replace("re:", '').replace("aw:", '').replace("fw:", '').replace(
                        "fwd:", '').replace('re[2]:', '').replace('re (3):', '').replace('(fwd)', '').strip()
                    subject = ' '.join([item for item in subject.split(' ') if item]).strip()
                    if not subject:
                        break
                    if not model1.search(subject):
                        break
                    if model2.search(subject):
                        break
                    if subject not in subjects:
                        subjects[subject] = []
                    subjects[subject].append(file)
                    break
    subjects = {subject: subjects[subject] for subject in subjects if len(subjects[subject]) > 2}
    print(f"======{len(subjects)} subjects in total======")
    # there should be 15183 subjects
    with open("subjects.txt", 'w') as f:
        f.write('\n'.join(sorted([subject for subject in subjects])))
    with open("subjects.json", 'w') as f:
        json.dump(subjects, f)


def extract_threads():
    print("====Extract threads and save to W3C.json====")
    with open("subjects.json", 'r') as f:
        subjects = json.load(f)
    threads = {}
    lens = []
    for subject in tqdm(subjects):
        thread = []
        for file in subjects[subject]:
            with codecs.open(f"w3c-emails/{file}", 'r', encoding='utf-8', errors='ignore')as f:
                email = {"to":[], "content": []}
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("docno="):
                        email["docno"] = line.split('=')[1].replace('"', '').strip()
                    elif line.startswith("received="):
                        email["received"] = line.split('=')[1].replace('"', '').strip()
                        items = [item for item in email["received"].split(' ') if item]
                        if items[-1] in ["EST", "EDT"]:
                            items = items[:-1]
                        items = items[:3] + [''.join(items[3:-1]), items[-1]]
                        date = ' '.join(items)
                        email["received_time"] = str(datetime.datetime.timestamp(datetime.datetime.strptime(date, "%c")))
                    elif line.startswith("isoreceived="):
                        email["isoreceived"] = line.split('=')[1].replace('"', '').strip()
                    elif line.startswith("name="):
                        email["name"] = line.split('=')[1].replace('"', '').strip()
                    elif line.startswith("email="):
                        email["email"] = line.split('=')[1].replace('"', '').strip()
                    elif line.startswith("subject="):
                        email["subject"] = line.split('=')[1].replace('"', '').strip()
                    elif line.startswith("To:") or line.startswith("to:") or line.startswith("TO:"):
                        email["to"].extend(line.replace("To:", '').replace("to:", '').replace("TO:", '').replace(
                            '"', '').replace("'", '').strip().split(','))
                    elif line.startswith("CC:") or line.startswith("Cc:") or line.startswith("cc:"):
                        email["to"].extend(line.replace("CC:", '').replace("Cc:", '').replace("cc:", '').replace(
                            '"', '').replace("'", '').strip().split(','))
                    elif line.startswith("sent=") or line.startswith("isosent=") or line.startswith("id=") \
                            or line.startswith("charset=") or line.startswith("expires=") \
                            or line.startswith("inreplyto="):
                        continue
                    else:
                        email["content"].append(line)
                try:
                    assert "received_time" in email
                    assert len(email["to"]) > 0
                except:
                    continue
                thread.append(email)

        if len(thread) == 0:
            continue

        thread = sorted(thread, key=lambda x: float(x["received_time"]))

        new_thread = []
        emails = {}
        persons = set()
        for email in thread:
            my = email["email"].lower()
            tos = [my]
            key = ' '.join(tos + [email["received_time"]])
            if key in emails:
                continue
            for to in email["to"]:
                if '<' in to and '>' in to:
                    to = to.split('<')[1].split('>')[0].lower()
                to = re.sub(r'\(\S+\)', '', to)
                tos.append(to.lower())
            if persons and len(set(tos) & persons) == 0:
                break
            new_thread.append(email)
            persons.update(tos)
            emails[key] = email

        content = set()
        for email in new_thread:
            content.add(' '.join(email["content"]).lower())
        if len(content) == 1:
            continue

        if 2 < len(new_thread) <= 50:
            lens.append(len(new_thread))
            threads[subject] = new_thread

    print(f"======{len(lens)} threads in total, the average/max thread lengths are {np.mean(lens)}/{max(lens)}======")
    # it should print 13794 threads in total, the average/max thread lengths are 6.445411048281862/50"
    with open("W3C.json", 'w') as f:
        json.dump(threads, f)


if __name__ == '__main__':
    unzip_files()
    group_emails_by_subjects()
    extract_threads()