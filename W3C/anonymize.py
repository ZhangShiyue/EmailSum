import os
import re
from tqdm import tqdm
import ujson as json
from nltk import word_tokenize
import nltk
from nltk.corpus import wordnet
import spacy
nlp = spacy.load('en_core_web_sm')


def remove_char(text):
    removelist = ' '
    text = re.sub(r'[^\w' + removelist + ']', '', text.lower())
    return text


def remove_email(text):
    subtext = text.split(' ')
    sts = []
    for i, st in enumerate(subtext):
        st = st.strip()
        st = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", f'USERNAME@DOMAIN.COM', st, flags=re.MULTILINE)
        sts.append(st)
    return ' '.join(sts)


def remove_url(text):
    subtext = text.split(' ')
    sts = []
    for st in subtext:
        st = re.sub(r'https?:\/\/.*[\r\n]*', 'HTTP://LINK', st, flags=re.MULTILINE)
        st = re.sub(r'www.*[\r\n]*', 'HTTP://LINK', st, flags=re.MULTILINE)
        sts.append(st)
    return ' '.join(sts)


def remove_ip_address(text):
    subtext = text.split(' ')
    sts = []
    for st in subtext:
        st = re.sub(r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}', 'IPADDRESS', st, flags=re.MULTILINE)
        st = re.sub(r'(?:[0-9]{1,3}\.){3}X', 'IPADDRESS', st, flags=re.MULTILINE)
        st = re.sub(r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})|([0-9a-fA-F]{4}\\.[0-9a-fA-F]{4}\\.[0-9a-fA-F]{4})', 'MACADDRESS', st, flags=re.MULTILINE)
        sts.append(st)
    return ' '.join(sts)


def remove_phone(text):
    st = re.sub(r"([2-9]\d{2}-\d{3}-\d{4})", 'PHONENUMBER', text, flags=re.MULTILINE)
    st = re.sub(r"([2-9]\d{2}-\d{3}-\d{3})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"([2-9]\d{2}- \d{3}-\d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"([2-9]\d{2} \d{3} \d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"([2-9]\d{2} \d{3}-\d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\([2-9]\d{2}-\d{3}-\d{4}\))", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"([2-9]\d{2}-\d{3}-\d{4},)", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"([2-9]\d{2}-\d{3}-\d{4}.)", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"([2-9]\d{2}/\d{3}-\d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"([2-9]\d{2}\.\d{3}\.\d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\([2-9]\d{2}\) \d{3}-\d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\([2-9]\d{2}\)\d{3}-\d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\([2-9]\d{2}\) \d{3} \d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\([2-9]\d{2}\) \d{3} \d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\d{2} \d{5} \d{5})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\d{2}-\d{3}-\d{7})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\d{2}\.\d{3}\.\d{7})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\d{2}-\d{3}-\d{6})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\d{2}-\d{1}-\d{4}-\d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\d{2}-\d{1}-\d{8})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\d{3}-\d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\d{5} \d{5})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\d{5} \d{6})", 'PHONENUMBER', st, flags=re.MULTILINE)
    st = re.sub(r"(\d{3}\.\d{4})", 'PHONENUMBER', st, flags=re.MULTILINE)
    for number in ["4085628026", "4084648998", "90-5334253539", "4088294796"]:
        if number in st:
            st = st.replace(number, 'PHONENUMBER')
    return st


def remove_local_path(text, path_dict):
    subtext = text.split(' ')
    sts = []
    for st in subtext:
        st = st.strip()
        if st.startswith("CONNECTION="):
            st = 'CONNECTION="DBPATH"'
        elif st.startswith("DB_URL="):
            st = 'DB_URL="DBPATH"'
        elif st in path_dict:
            st = path_dict[st]
        sts.append(st)
    return ' '.join(sts)


def remove_last_names(text):
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary=False)
    person_list = []
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1:  # avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []
    names = person_list.copy()
    for person in person_list:
        person_split = person.split(" ")
        if len(person_split) == 2:
            if "Nunn" in person_split or "Chan" in person_split or "Richard" in person_split:
                continue
        for name in person_split:
            if wordnet.synsets(name):
                if (name in person):
                    names.remove(person)
                    break
    for name in names:
        words = name.split(' ')
        text = text.replace(name, words[0].strip() + "'s" if "'s" in words[-1] else words[0].strip())
    return text


def remove_last_name_in_context(text, name_dict):
    error_words = ["To", "Update", "Hello", "Hi", "User", "ASAP", "Enterprise", "Screen", "DB", "Chart",
             "hai", "Hey", "Box", "Bug", "Time", "Greetings", "Bus", "R&D", "HP", "Doc",
             "Hai", "Apps", "Software", "and", "7th", "IE", "Schedule", "delete", "Dev",
             "Worksheet", "Name", "Girl", "Appliance", "BEFORE", "Encyclopedic", "Topics", "Configure",
             "Release", "office", "Alerts", "Installer",
             "Admin", "BBQ", "Mtg", "TBD", "Owner", "data", "app", "Interface", "Auto",
             "Adpater", "Assign", "Boxer", "Ask", "Objectives", "Discuss", "logins", "Code", "Alternative",
             "xml", "Comments"]
    doc = nlp(text)
    ents = doc.ents
    for ent in ents:
        if ent.label_ != 'PERSON':
            continue
        ent_text = ent.text.strip()
        ent_words = [word.strip() for word in ent_text.split(' ') if word.strip()]
        if len(ent_words) == 1:
            continue
        if set(ent_words) & set(error_words):
            continue
        if ent_text in name_dict:
            text = text.replace(ent_text, name_dict[ent_text])
    return text


def filter():
    def _name_dict():
        name_dict = {}
        with open(f"../Avocado/anonymize_files/person_names.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(',')
                if len(items) > 1:
                    if items[0].strip() != items[1].strip():
                        name_dict[items[0].strip()] = items[1].strip()
        return name_dict

    def _subject_filter(subject):
        model = re.compile(r'[a-z]')
        if subject[0] == "#" or "no subject" in subject or subject[0] == "$" or "***********" in subject \
                or "re:" in subject or "autoreply" in subject or "confidential" in subject:
            return True
        if not model.search(subject):
            return True
        return False

    def _thread_filter(thread):
        end_words = ['cheers', 'regards', 'thanks', 'best', 'kind regards', 'best regards',
                     "thanks and looking forward", "super thanks"]
        contents = []
        total_words = []
        froms = []
        if not thread[0]['subject'] or "re:" in thread[0]['subject'].lower() or \
                "fw:" in thread[0]['subject'].lower() or "fwd:" in thread[0]['subject'].lower() or \
                "fw:" in thread[1]['subject'].lower() or "fwd:" in thread[1]['subject'].lower():
            return []
        for email in thread:
            next_signature = False
            if not email['name'].strip().split(' ')[0].split('@')[0].split('.')[0]:
                return []
            names = email['name'].strip().lower().split()
            froms.append(f"{email['name'].strip().lower()}")
            full_names = []
            for i in range(1, len(names) + 1):
                for j in range(len(names) - i + 1):
                    full_names.append(' '.join(names[j: j + i]))
            lines = email["content"]
            new_lines = []
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('>') or line.startswith("Message-ID:") or line.startswith("From:") \
                        or line.startswith('In-reply-to:') or "wrote:" in line or line.startswith("Subject:") \
                        or line.startswith('Date:'):  # special for W3C
                    continue
                line = line.replace("-->", ' ').strip()
                clean_line = remove_char(line)
                if not line:
                    continue
                if "*****************" in line or "-------------------" in line or '_______________' in line or \
                        '~~~~~~~~~~~~~~~~~' in line or "============" in line or \
                        "-----Original Appointment-----" in line:
                    break
                if line == "---" or line == "--" or line == "Manager" or line == "R&D":
                    break
                # signature
                if clean_line in full_names:
                    new_lines.append(line.split(' ')[0].strip())
                    break
                tokens = clean_line.split(' ')
                if len(tokens) == 3 and (tokens[0] in full_names or tokens[1] in full_names or tokens[2] in full_names):
                    new_lines.append(line.split(' ')[0].strip())
                    break
                if line.startswith('Address:'):
                    if line.split(':')[1].strip():
                        line = 'Address: ADDRESS'
                elif line.startswith("Contract #:"):
                    if line.split(':')[1].strip():
                        line = 'Contract #: CONTRACTNUMBER'
                line = remove_email(line)
                line = remove_url(line)
                line = remove_ip_address(line)
                line = remove_phone(line)
                line = remove_last_names(line)
                line = remove_last_name_in_context(line, name_dict)
                line = remove_local_path(line, {})
                if next_signature:
                    new_lines.append(line.split(' ')[0])
                    break
                else:
                    new_lines.append(line)
                if clean_line in end_words:
                    next_signature = True
            new_content = '\n'.join(new_lines)
            if "password" in new_content or "passcode" in new_content or "pw:" in new_content or \
                "Password" in new_content or "Passcode" in new_content or "PW:" in new_content or \
                    "passwd" in new_content or "Pass code" in new_content or "pass code" in new_content:
                return []
            words = word_tokenize(new_content)
            total_words.extend(words)
            if len(words) > 200 or len(words) < 5:
                return []
            contents.append(new_content)
        if len(set(contents)) == 1:
            return []
        if len(set(froms)) == 1:
            return []
        if len(total_words) > 1000 or len(total_words) < 30:
            return []
        return contents

    def _distinguish_names(thread):
        names = []
        first_names = {}
        for email in thread:
            from_name = email["name"].strip()
            to_names = [person.split('<')[0].strip() for person in email["to"]]
            for name in [from_name] + to_names:
                if name not in names:
                    names.append(name)
                    first_name = name.split(' ')[0].split('@')[0].split('.')[0]
                    if first_name not in first_names:
                        first_names[first_name] = 0
                    else:
                        first_names[first_name] += 1
        count = {}
        name_map = {}
        for name in names:
            first_name = name.split(' ')[0].split('@')[0].split('.')[0]
            if first_names[first_name] > 0:
                if first_name not in count:
                    count[first_name] = 0
                else:
                    count[first_name] += 1
                name_map[name] = first_name + f'-{count[first_name]}'
            else:
                name_map[name] = first_name
        return name_map

    with open("W3C.json", 'r') as f:
        data = json.load(f)
    name_dict = _name_dict()

    if not os.path.exists("W3C_threads"):
        os.mkdir("W3C_threads")

    data = sorted(data.items(), key=lambda x: x[0])
    for threadno, (subject, thread) in enumerate(tqdm(data)):
        if _subject_filter(subject.lower()):
            continue
        content_lengths = [len(email["content"]) for email in thread]
        if len(content_lengths) > 10:
            continue
        directory = 7 if len(content_lengths) >= 7 else len(content_lengths)
        new_contents = _thread_filter(thread)
        if not new_contents:
            continue
        if not os.path.exists(f"W3C_threads/{directory}"):
            os.mkdir(f"W3C_threads/{directory}")
        if os.path.exists(f"W3C_threads/{directory}/{threadno}"):
            # clean existing file
            with open(f"W3C_threads/{directory}/{threadno}", 'w') as f:
                f.write("")
        name_map = _distinguish_names(thread)
        with open(f"W3C_threads/{directory}/{threadno}", 'a') as f:
            f.write(f"THREAD {threadno}\n\n"
                    f"subject: {subject}\n"
                    f"# of emails: {len(content_lengths)}\n\n")
            for j, email in enumerate(thread):
                form_first_name = name_map[email['name'].strip()]
                to_first_names = [name_map[person.split('<')[0].strip()] for person in email['to']]
                f.write(f"Email {j}\n"
                        f"From: {form_first_name}\n"
                        f"To: {', '.join(to_first_names)}\n\n"
                        f"Content:\n {new_contents[j]}\n\n"
                        f"==============================================================\n\n")


if __name__ == '__main__':
    filter()