import os
import re
from tqdm import tqdm
import ujson as json
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
from nltk.corpus import wordnet
# from langdetect import detect
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
    for subtree in sentt.subtrees(filter=lambda t: t.label()  == 'PERSON'):
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
        with open("anonymize_files/person_names.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(',')
                if len(items) > 1:
                    if items[0].strip() != items[1].strip():
                        name_dict[items[0].strip()] = items[1].strip()
        return name_dict

    def _path_dict():
        path_dict = {}
        with open("anonymize_files/paths.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                st = line.strip()
                new_st = []
                for t_item in st.split('\t'):
                    new_t_item = []
                    for s_item in t_item.split(';'):
                        new_s_item = []
                        for c_item in s_item.split(','):
                            new_c_item = []
                            for e_item in c_item.split('='):
                                new_e_item = []
                                for cc_item in e_item.split(':'):
                                    for spliter in ['\\', '/']:
                                        if spliter in cc_item:
                                            last = cc_item.split(spliter)[-1]
                                            sec_last = cc_item.split(spliter)[-2]
                                            cc_item = f"PATH{spliter}{last if last else sec_last}"
                                    new_e_item.append(cc_item)
                                e_item = ':'.join(new_e_item)
                                new_c_item.append(e_item)
                            c_item = '='.join(new_c_item)
                            new_s_item.append(c_item)
                        s_item = ','.join(new_s_item)
                        new_t_item.append(s_item)
                    t_item = ';'.join(new_t_item)
                    new_st.append(t_item)
                new_st = '\t'.join(new_st)
                path_dict[st] = new_st
        return path_dict

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
            for line in lines:
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
                if line == "AvocadoIT, Inc." or line == "AvocadoIT  Inc.,"  or line == "AvocadoIT, Inc" \
                        or line == "AvocadoIT" or line == "AvocadoIT, Your Business Everywhere!" or \
                        line == "AvocadoIT Canada Inc." or line == "Director of Corporate Marketing" or \
                        line == "Director of Business Development" or line.startswith('Assit.') or \
                        line.startswith('office') or line.startswith('fax') or \
                        line.startswith('Office') or line.startswith('Cell') or \
                        "Enabling Infrastructure for Post-PC Era" in line or line == "Manager, R&D," or \
                        line == "Robert, Global Alliance Manager" or line == "Global Alliance Manager" or \
                        line == "Sr. Program Manager" or line == "John, District Manager" or line == "Facility Manager" \
                        or line == "Chief Executive Officer" or line == "Director of Sales" \
                        or line == "David Chan, VP Business Development" or line == "avocadoit.com" or \
                        line == "Sent wirelessly from AvocadoIT, Inc." or line == "Your Business  Everywhere" or \
                        line == "Director of Networks and Operations":
                    break
                if line.startswith('Address:'):
                    if line.split(':')[1].strip():
                        line = 'Address: ADDRESS'
                elif line.startswith("Contract #:"):
                    if line.split(':')[1].strip():
                        line = 'Contract #: CONTRACTNUMBER'
                for address in ["430 Cahuenga Blvd. Suite #128 Universal City, CA 91602", "2211 N Ist Street",
                                "105-0001 4-3-20 Toranomon", "Kamiyacho Mori Building 14F", "4050 Legato Rd.",
                                "Fairfax, VA  22032", "Hewlett Packard de Mexico", "S.A. de C.V.",
                                "Prolongacion Reforma 700", "Col. Lomas de Santa Fe", "Mexico City, MEXICO  01210",
                                "1359 Palos Verdes, San Mateo, CA 94403", "Hewlett-Packard", "371 Hoes Lane",
                                "Piscataway, NJ  08854", "8745 W. Higgins, Ste 360", "Chicago, I'll 60631",
                                "550 Madison Avenue", "4615 Winding Way", "CA - 95129", "San Jose, CA 95131",
                                "5535 Woodlawn N", "SAN JOSE, CA 95131", "2211 North First Street",
                                "Seattle, WA  98103", "471 Santa Rosa Drive", "Los Gatos, CA 95032", "3231 scott",
                                "santa clara, ca, 95054", "430 Cahuenga", "Suite #128 Universal City, CA 91602",
                                "49 Ontario", "260 King Street", "2211 N 1st Street", "430 kipling ave in Palo alto",
                                "51 B Harbor", "East Hampton NY 11937", "60 State Street", "1230 Ave of Americas",
                                "222 Mason Street", "San Francisco, CA 94102", "2211 N. FIRST STREET",
                                "2052 Joel Court, Palm Harbor, FL 34683", "Silicon Valley Conference Center",
                                "55 Broad Street", "120 Adelaide Street West"]:
                    if address in line:
                        line = line.replace(address, 'ADDRESS')
                for direction in ["CA-92", "HAYWARD/SAN MATEO BR/HALF MOON BAY", "HILLSDALE BOULEVARD", "W HILLSDALE",
                                  "GLENDORA", "PALOS VERDES"]:
                    if direction in line:
                        line = line.replace(direction, 'DIRECTION')
                for tracking_number in ["828839908675", "8320709434", "03001295000057260063", "8238 5389 8547"]:
                    if tracking_number in line:
                        line = line.replace(tracking_number, 'TRACKINGNUMBER')
                for account_number in ["04488565", "09352584", "09684507", "09924713", "10209765", "R116086",
                                       "00067 244 109", "15736722", "467 223 912", "1958 0551", "00 774 701 124",
                                       "152990642", "BJ422463", "55615206"]:
                    if account_number in line:
                        line = line.replace(account_number, 'ACCOUNTNUMBER')
                for confirmation_number in ["119914", "2726335"]:
                    if confirmation_number in line:
                        line = line.replace(confirmation_number, 'CONFIRMATIONNUMBER')
                for card_number in ["5322 6915 1892 4823"]:
                    if card_number in line:
                        line = line.replace(card_number, 'CARDNUMBER')
                for case_number in ["WAC-01-033-50034"]:
                    if case_number in line:
                        line = line.replace(case_number, 'CASENUMBER')
                for building_number in ["53849"]:
                    if building_number in line:
                        line = line.replace(building_number, 'BUILDINGNUMBER')
                for conference_number in ["1-888-7428686"]:
                    if conference_number in line:
                        line = line.replace(conference_number, 'CONFERENCENUMBER')
                for order_number in ["N44181683", "N44181683", "N31634945"]:
                    if order_number in line:
                        line = line.replace(order_number, 'ORDERNUMBER')
                for number in ["0.-1723910PHONENUMBER8638", "167018e9", "11050333", "102306"]:
                    if number in line:
                        line = line.replace(number, 'NUMBER')
                for person_name, person_fname in [("Krishna mohan's", "Krishna's"), ("Tom Trenga", "Tom"),
                                                  ("David Chan", "David"), ("Anne-Christine", "Anne"),
                                                  ("anne-christine", "anne"), ("Anne-christine", "Anne"),
                                                  ("Troy Fernwalt", "Troy"), ("Ricardo Garcia", "Ricardo"),
                                                  ("Mehrak Hamzeh", "Mehrak"), ("Marc Friend's", "Marc's"),
                                                  ("Shashi Vitthal", "Shashi"), ("Kenneth Loveless", "Kenneth"),
                                                  ("Harry Kellog", "Harry"), ("Ming Guo", "Ming"),
                                                  ("Fernand Braganza", "Fernand"), ("Nilesh Bodade", "Nilesh")]:
                    if person_name in line:
                        line = line.replace(person_name, person_fname)
                line = remove_email(line)
                line = remove_url(line)
                line = remove_ip_address(line)
                line = remove_phone(line)
                line = remove_last_names(line)
                line = remove_last_name_in_context(line, name_dict)
                line = remove_local_path(line, path_dict)
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
        # remove language detection!!! because we find it is not accurate and causes good threads to be filtered
        # for cont in contents:
        #     if detect(cont) != "en":
        #         return []
        return contents

    def _distinguish_names(thread):
        names = []
        first_names = {}
        for email in thread:
            from_name = email["name"].replace('(E-mail)', '').replace('(E-Mail)', '').replace("'", '').replace(
                'Breen, Andrew', 'Andrew Breen').replace('- India', '').replace(
                'Keith Ritchie2', 'Keith Ritchie').replace(":", '').replace('(Toronto)', '').strip()
            to_names = [person.split('<')[0].replace('(E-mail)', '').replace('(E-Mail)', '').replace("'", '').replace(
                'Breen, Andrew', 'Andrew Breen').replace('- India', '').replace(
                'Keith Ritchie2', 'Keith Ritchie').replace(":", '').replace('(Toronto)', '').strip() for person in email["to"]]
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

    with open("Avocado.json", 'r') as f:
        data = json.load(f)
    name_dict = _name_dict()
    path_dict = _path_dict()

    if not os.path.exists("Avocado_threads"):
        os.mkdir("Avocado_threads")

    data = sorted(data.items(), key=lambda x: x[0])
    for i, (subject, thread) in enumerate(tqdm(data)):
        if _subject_filter(subject.lower()):
            continue
        content_lengths = [len(email["content"]) for email in thread]
        if len(content_lengths) > 10:   # thread length <= 10
            continue
        directory = 7 if len(content_lengths) >= 7 else len(content_lengths)
        new_contents = _thread_filter(thread)
        if not new_contents:
            continue
        name_map = _distinguish_names(thread)
        if not os.path.exists(f"Avocado_threads/{directory}"):
            os.mkdir(f"Avocado_threads/{directory}")
        if os.path.exists(f"Avocado_threads/{directory}/{i}"):
            # clean existing file
            with open(f"Avocado_threads/{directory}/{i}", 'w') as f:
                f.write("")
        with open(f"Avocado_threads/{directory}/{i}", 'a') as f:
            f.write(f"THREAD {i}\n\n"
                    f"subject: {subject}\n"
                    f"# of emails: {len(content_lengths)}\n\n")
            for j, email in enumerate(thread):
                form_first_name = name_map[email['name'].replace('(E-mail)', '').replace('(E-Mail)', '').replace("'", '').replace(
                    'Breen, Andrew', 'Andrew Breen').replace('- India', '').replace(
                    'Keith Ritchie2', 'Keith Ritchie').replace(":", '').replace('(Toronto)', '').strip()]
                if form_first_name in ["4088958871"]:
                    form_first_name = "PHONENUMBER"
                to_first_names = [name_map[person.split('<')[0].replace('(E-mail)', '').replace('(E-Mail)', '').replace("'", '').replace(
                    'Breen, Andrew', 'Andrew Breen').replace('- India', '').replace(
                    'Keith Ritchie2', 'Keith Ritchie').replace(":", '').replace('(Toronto)', '').strip()] for person in email['to']]
                for i, to_first_name in enumerate(to_first_names):
                    if to_first_name in ["4088958871"]:
                        to_first_names[i] = "PHONENUMBER"
                f.write(f"Email {j}\n"
                        f"From: {form_first_name}\n"
                        f"To: {', '.join(to_first_names)}\n\n"
                        f"Content:\n {new_contents[j]}\n\n"
                        f"==============================================================\n\n")


def print_out():
    with open("Avocado.json", 'r') as f:
        data = json.load(f)
    data = sorted(data.items(), key=lambda x: x[0])
    for i, (subject, thread) in enumerate(data):
        if i == 7275:
            content_lengths = [len(email["content"]) for email in thread]
            print(f"subject: {subject}\n"
                  f"# of emails: {len(content_lengths)}\n\n")
            for i, email in enumerate(thread):
                content = '\n'.join(email['content'])
                print(f"Email {i}\n"
                      f"Date: {email['date']}\n"
                      f"From: {email['name']}, {email['email']}\n"
                      f"Subject: {email['subject']}\n"
                      f"To: {','.join(email['to'])}\n\n"
                      f"Content:\n {content}\n\n"
                      f"==============================================================\n\n")


if __name__ == '__main__':
    filter()