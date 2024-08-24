# text_segment_cross_segment_attention
Text segment using cross segment attention

## Dataset

使用基于 wikipedia 的中文数据集进行训练

来自于[Wikipedia monolingual corpora](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/)

| Wikipedia XML file             | unzipped file size | language | number of articles | number of paragraphs | number of tokens |
| ------------------------------ | ------------------ | -------- | ------------------ | -------------------- | ---------------- |
| zhwiki-20181001-corpus.xml.bz2 | 4.1G               | Chinese  | 1,619,172          | 4,853,082            | 54,747,370       |

## Cross Segment Attention

该方法来自于 [《Text Segmentation by Cross Segment Attention》](https://arxiv.org/abs/2004.14535)