import os
import re
import codecs
import datetime
import numpy as np
import ujson as json
import xml.etree.ElementTree as ET
from tqdm import tqdm


def unzip_files():
    files = os.listdir(f"{ROOT_DIR}/LDC2015T03/Data/avocado-1.0.2/data/text/")
    for file in files:
        os.system(f"unzip {ROOT_DIR}/LDC2015T03/Data/avocado-1.0.2/data/text/{file} "
                  f"-d avocado_text")


def get_emails():
    print("====Get all emails and save to emails.json====")
    files = os.listdir(f"{ROOT_DIR}/LDC2015T03/Data/avocado-1.0.2/data/custodians")
    emails = {}
    dedup_emails = set()
    for file in tqdm(files):
        parsedXML = ET.parse(f"{ROOT_DIR}/LDC2015T03/Data/avocado-1.0.2/data/custodians/{file}")
        custodian = parsedXML.getroot()
        custodian = {item.tag: item for item in custodian}
        items = custodian["items"]
        for item in items:
            subitems = {subitem.tag: subitem for subitem in item}
            if "files" not in subitems or "metadata" not in subitems:
                continue
            for file in subitems["files"]:
                path = file.attrib['path']
                items = path.split('/')
                if items[0] == "text" and 'EM' in items[2]:
                    filename = items[2]
                    break
            emails[filename] = {}
            dedup_emails.add(filename)
            if "relationships" in subitems:
                for relation in subitems["relationships"]:
                    if relation.tag == "duplicate_of":
                        dedup_emails.add(f'{relation.attrib["id"]}.txt')
                    elif relation.tag == "reply_to":
                        emails[filename]["reply_to"] = f'{relation.attrib["id"]}.txt'
            for field in subitems["metadata"]:
                if field.attrib["name"] == "arrival_date":
                    emails[filename]["date"] = field.text
                elif field.attrib["name"] == "outlook_sender_name":
                    emails[filename]["sender"] = field.text
                elif field.attrib["name"] == "processed_subject":
                    emails[filename]["processed_subject"] = field.text
                elif field.attrib["name"] == "subject":
                    emails[filename]["subject"] = field.text
    print(f"======{len(emails)} emails in total======")
    # there should be 937958 emails
    # !!! Please use our provided emails.json to get the exact threads we used because the order of emails matters.
    with open("emails_new.json", 'w') as f:
        json.dump(emails, f)


def group_emails_by_subjects():
    print("====Group emails by their subjects and save to subject.json====")
    subjects = {}
    # !!! Please use our provided emails.json to get the exact threads we used because the order of emails matters.
    with open("emails.json", 'r') as f:
        emails = json.load(f)
    model1 = re.compile(r'[a-zA-Z0-9]')
    model2 = re.compile(r'[^\x00-\x7f]')
    for email in tqdm(emails):
        directory = email.split('-')[0]
        with codecs.open(f"avocado_text/{directory}/{email}", 'r', encoding='utf-8', errors='ignore')as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("Received:"):
                    break
                if line.startswith("Subject:"):
                    subject = line.replace("Subject:", '').lower()
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
                    subjects[subject].append(email)
                    break
    subjects = {subject: subjects[subject] for subject in subjects if len(subjects[subject]) > 2}
    print(f"======{len(subjects)} subjects in total======")
    # there should be 51367 subjects
    with open("subjects.txt", 'w') as f:
        f.write('\n'.join(sorted([subject for subject in subjects])))
    with open("subjects.json", 'w') as f:
        json.dump(subjects, f)


def extract_threads():
    print("====Extract threads and save to Avocado.json====")
    threads = {}
    lens = []
    with open("subjects.json", 'r') as f:
        subjects = json.load(f)
    datestr = "%d %b %Y %H:%M:%S UTC"
    for subject in tqdm(subjects):
        thread = []
        for file in subjects[subject]:
            name = None
            email = None
            sub = None
            date = None
            received_time = None
            tos = []
            content = []
            directory = file.split('-')[0]
            with open(f"avocado_text/{directory}/{file}", 'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    elif "-----Original Message-----" in line or '-----Ursprüngliche Nachricht-----' in line:
                        break
                    elif line.startswith("From:"):
                        items = line.replace("From:", '').split('<')
                        if len(items) < 2:
                            break
                        email = items[1].replace('>', '').strip()
                        name = items[0].replace('"', '').strip()
                    elif (line.startswith("To:") or line.startswith("to:") or line.startswith("TO:")):
                        tos.extend(line.replace("To:", '').replace("to:", '').replace("TO:", '').replace(
                            '"', '').replace("'", '').strip().split(','))
                    elif (line.startswith("CC:") or line.startswith("Cc:") or line.startswith("cc:")):
                        tos.extend(line.replace("CC:", '').replace("Cc:", '').replace("cc:", '').replace(
                            '"', '').replace("'", '').strip().split(','))
                    elif (line.startswith("Bcc:") or line.startswith("bcc:") or line.startswith("BCC:")):
                        tos.extend(line.replace("Bcc:", '').replace("bcc:", '').replace("BCC:", '').replace(
                            '"', '').replace("'", '').strip().split(','))
                    elif line.startswith("Subject:"):
                        sub = line.replace("Subject:", '').strip()
                    elif line.startswith("Date:"):
                        try:
                            date = line.replace("Date:", '').strip()
                            received_time = str(datetime.datetime.timestamp(datetime.datetime.strptime(date, datestr)))
                        except:
                            break
                    elif (line.startswith("Message-ID:") or
                            line.startswith("In-Reply-To:") or line.startswith("MIME-Version:") or
                            line.startswith("Reply-To:") or line.startswith("Content-Type:") or
                            line.startswith("Sender:")):
                        continue
                    elif "===============================" in line:
                        continue
                    else:
                        content.append(line)
            if not email or not tos or not content or not date or not received_time:
                continue
            else:
                # received_time = str(time.mktime(datetime.datetime.strptime(date, datestr).timetuple()))
                thread.append({"file": file, "name": name, "email": email, "subject": sub, "date": date,
                               "received_time": received_time, "to": tos, "content": content})
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

        # remove repeat content email threads
        content = set()
        for email in new_thread:
            content.add(' '.join(email["content"]).lower())
        if len(content) == 1:
            continue

        if 2 < len(new_thread) <= 50:   # 2 < thread length <= 50
            lens.append(len(new_thread))
            threads[subject] = new_thread

    print(f"======{len(lens)} threads in total, the average/max thread lengths are {np.mean(lens)}/{max(lens)}======")
    # it should print "28416 threads in total, the average/max thread lengths are 5.34667088963964/50"
    with open("Avocado.json", 'w') as f:
        json.dump(threads, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./", type=str, help="the directory contains LDC2015T03")
    args = parser.parse_args()
    ROOT_DIR = args.root_dir

    if not os.path.exists("avocado_text"):
        os.mkdir("avocado_text")
    unzip_files()
    group_emails_by_subjects()
    extract_threads()
