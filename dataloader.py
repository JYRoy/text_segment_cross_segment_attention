from torch.utils.data import Dataset, DataLoader
import re
from copy import deepcopy


class WikipediaDataset(Dataset):

    def __init__(self, file):
        self.root = file

    def para2sents(self, paragraph):
        for sent in re.findall("[^!?。\.\!\?]+[!?。\.\!\?]?", paragraph, flags=re.U):
            yield sent

    def process_xml(self, out_file="./datasets/wiki-zh.txt"):
        with open(self.root, encoding="utf-8") as f:
            text = f.read()

        root_pattern = "(?<=\<{0}>\n)[\s\S]*?(?=\n</{0}>)"
        para_pattern = root_pattern.format("p")
        headline_pattern = root_pattern.format("h")
        para_matches = list(re.finditer(re.compile(para_pattern), text))
        paras = list()
        para_spans = list()
        for para_m in para_matches:
            paras.append(para_m.group())
            para_spans.append(para_m.span())

        para_info = list(zip(paras, para_spans))

        ind_headline_info = list()
        headline_matches = list(re.finditer(re.compile(headline_pattern), text))
        if headline_matches:
            headlines = list()
            headline_spans = list()
            for headline_m in headline_matches:

                headlines.append(headline_m.group())
                headline_spans.append(headline_m.span())

            headline_info = list(zip(headlines, headline_spans))

            for h_info in headline_info:
                h_start, h_end = h_info[1]
                in_para = False
                for p_info in para_info:
                    p_start, p_end = p_info[1]
                    if p_start <= h_start and h_end <= p_end:
                        in_para = True
                        # logger.info('headline in para ...')
                if not in_para:
                    ind_headline_info.append(h_info)

        sorted_items = deepcopy(list(para_info))
        if ind_headline_info:
            for ind_h_info in ind_headline_info:
                ind_h_start = ind_h_info[1][0]
                p_span_starts = [p_info[1][0] for p_info in para_info]
                insert_idx = None
                for idx, p_span_start in enumerate(p_span_starts):
                    if ind_h_start < p_span_start:
                        insert_idx = idx
                        break
                item_dict = {"content": ind_h_info[0], "is_headline": True}

                sorted_items.insert(insert_idx, item_dict)

        for idx, item in enumerate(sorted_items):
            if type(item) != dict:
                item_dict = {"content": item[0], "is_headline": False}
                sorted_items[idx] = item_dict

        for idx, item in enumerate(sorted_items):
            if item["is_headline"]:
                continue

            headline_matches = list(
                re.finditer(re.compile(headline_pattern), item["content"])
            )

            if not headline_matches:
                continue

            new_items = list()
            for headline_m in headline_matches:
                inner_headline_item = {
                    "content": headline_m.group(),
                    "is_headline": True,
                }

                new_items.insert(0, inner_headline_item)

            rest_pattern = "(?<=\</h>\n)[\s\S]*"
            rest_match = re.search(re.compile(rest_pattern), item["content"])

            if rest_match:
                rest_para_item = {
                    "content": rest_match.group(),
                    "is_headline": False,
                }

                new_items.insert(0, rest_para_item)

            del sorted_items[idx]

            for new_item in new_items:
                sorted_items.insert(idx, new_item)

        proc_lines = list()

        content = self.para2sents(content)
        for element in sorted_items:
            element_lines = self.proc_content(**element)
            proc_lines.extend(element_lines)

        proc_lines.insert(0, "#s-doc")
        proc_lines.append("#e-doc")

        out_text = "\n".join(proc_lines)

        with open(out_file, mode="a", encoding="utf-8") as f:
            f.write(out_text)

    def proc_content(
        self, content, is_headline, use_sent_seg=False, convert2simple=False
    ):

        proc_lines = list()
        for sent in content:
            proc_lines.append("#s-sent")
            # proc_lines.append("\n".join(words))
            proc_lines.append(sent)
            proc_lines.append("#e-sent")

        if is_headline:
            proc_lines.insert(0, "#s-headline")
            proc_lines.append("#e-headline")
        else:
            proc_lines.insert(0, "#s-para")
            proc_lines.append("#e-para")

        return proc_lines

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.textfiles)


xml_file = "../dev/zhwiki-mini-corpus.xml"
train_dataset = WikipediaDataset(xml_file)
train_dataset.process_xml(out_file="./datasets/wiki-zh.txt")
