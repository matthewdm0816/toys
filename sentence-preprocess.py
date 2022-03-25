import re
from icecream import ic
import glob
from tqdm import tqdm

# from https://blog.csdn.net/blmoistawinde/article/details/82379256
# 版本为python3，如果为python2需要在字符串前面加上u
def cut_sent(para):
    para = re.sub("([。！？\?])([^”’])", r"\1\n\2", para)  # 单字符断句符
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)  # 英文省略号
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)  # 中文省略号
    para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


# filenames = ["神们自己.txt"]
filenames = glob.glob("ScifiChinese/*.txt")
ic(filenames)
bad_words = [
    "wulali",
    "乌拉科幻小说网",
    "=========",
    "本站小说仅供试阅",
    "】",
    "【",
    "注:",
    "--------",
]
sentences_total = []
for filename in tqdm(filenames):
    print(f"Processing {filename}")
    with open(filename, "r", encoding="gb18030") as f:
        text = f.read()
    lines = text.split("\n")
    # Remove chinese tab
    for idx, line in enumerate(lines):
        lines[idx] = re.sub("\u3000", "", line)
    # Remove empty line
    lines = [line for line in lines if len(line) > 0]

    # Split sentences into list
    sentences = sum(map(cut_sent, lines), start=[])
    # Remove empty sentence
    sentences = [sentence for sentence in sentences if len(sentence) > 0]
    # Remove sentences including bad words
    sentences = [
        sentence
        for sentence in sentences
        if re.search("|".join(bad_words), sentence, flags=re.IGNORECASE) is None
    ]
    ic(len(sentences))

    sentences_total.append(sentences)
ic(len(sentences_total))
ic(sum(map(len, sentences_total)))
# Save sentences
with open("UER-py/corpora/scifi.txt", "w", encoding="utf-8") as f:
    for sentences in sentences_total:
        for sentence in sentences:
            f.write(sentence + "\n")
        f.write("\n")
print("Done.")
