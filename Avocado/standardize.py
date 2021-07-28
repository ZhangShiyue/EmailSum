import os
import numpy as np
import ujson as json
import nltk
nltk.download('punkt')


def prepare_one(lines):
    thread_id = None
    subject = None
    numemail = 0
    froms = [''] * 10
    tos = [''] * 10
    emails = [''] * 10
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line:
            if line.startswith("THREAD"):
                thread_id = int(line.split(' ')[1])
            elif line.startswith("subject:"):
                subject = line.replace("subject:", "").strip()
                # if "docomo" in subject:
                #     print(thread_id)
                #     exit()
            elif line.startswith("# of emails:"):
                numemail = int(line.replace("# of emails:", "").strip())
            elif line.startswith("Email"):
                index = int(line.split(' ')[1].strip())
            elif line.startswith("From:"):
                froms[index] = line.replace('From:', '').strip()
            elif line.startswith("To:"):
                tos[index] = line.replace('To:', '').strip()
            elif line.startswith("Content:"):
                i += 1
                line = lines[i].strip()
                content = []
                while i < len(lines) and (not line.startswith("===")):
                    if line:
                        content.append(line)
                    i += 1
                    line = lines[i].strip()
                emails[index] = ' '.join(content)
        i += 1
    return thread_id, subject, numemail, froms, tos, emails


def standardize():
    with open("summaries/EmailSum_data.json", 'r') as f:
        data = json.load(f)["data"]
        train = data["train"]
        test = data["test"]
    with open("summaries/one_more_reference.json", 'r') as f:
        test1 = json.load(f)["test"]
    np.random.seed(42)
    np.random.shuffle(train)
    dev = train[:249]
    train = train[249:]

    if not os.path.exists("exp_data"):
        os.mkdir("exp_data")
    if not os.path.exists("exp_data/data_email_short"):
        os.mkdir("exp_data/data_email_short")
    if not os.path.exists("exp_data/data_email_long"):
        os.mkdir("exp_data/data_email_long")

    for dset, sset in [(train, "train"), (dev, "dev"), (test, "test")]:
        short_source, short_target, short_id = [], [], []
        long_source, long_target, long_id = [], [], []
        short_target1, long_target1 = [], []
        for thread in dset:
            thread_id = thread["thread_id"]
            numemail = thread["num_of_emails"]
            short_summary = thread["short_summary"]["content"]
            long_summary = thread["long_summary"]["content"]
            with open(f"Avocado_threads/{numemail}/{thread_id}", 'r') as f:
                lines = f.readlines()
            _, subject, _, froms, tos, emails = prepare_one(lines)
            source = '|||'.join([f"{f}: {e}" for f, e in zip(["Subject"] + froms, [subject] + emails) if f and e])
            short_source.append(source)
            short_target.append(short_summary)
            short_id.append(thread_id)
            long_source.append(source)
            long_target.append(long_summary)
            long_id.append(thread_id)
            if sset == "test":
                short_target1.append(test1[thread_id]["short_summary"]["content"])
                long_target1.append(test1[thread_id]["long_summary"]["content"])
        with open(f"exp_data/data_email_short/{sset}.source", 'w') as f:
            f.write('\n'.join(short_source))
        with open(f"exp_data/data_email_short/{sset}.target", 'w') as f:
            f.write('\n'.join(short_target))
        with open(f"exp_data/data_email_short/{sset}.id", 'w') as f:
            f.write('\n'.join(short_id))
        with open(f"exp_data/data_email_long/{sset}.source", 'w') as f:
            f.write('\n'.join(long_source))
        with open(f"exp_data/data_email_long/{sset}.target", 'w') as f:
            f.write('\n'.join(long_target))
        with open(f"exp_data/data_email_long/{sset}.id", 'w') as f:
            f.write('\n'.join(long_id))
        if sset == "test":
            with open(f"exp_data/data_email_short/{sset}.target1", 'w') as f:
                f.write('\n'.join(short_target1))
            with open(f"exp_data/data_email_long/{sset}.target1", 'w') as f:
                f.write('\n'.join(long_target1))


if __name__ == '__main__':
    standardize()