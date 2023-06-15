from enum import Enum

import pandas as pd

import numpy.typing as npt
import numpy as np

import os


# XXX What to do about newlines when processing?

class StrEnum(str, Enum):
    pass

class DataSets(StrEnum):
    chapters = "chapters.tsv"
    documents = "documents.tsv"
    sentences = "sentences.tsv"

REPO = "toki-pona-dataset"
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_dataset(datatype: StrEnum) -> pd.DataFrame:
    path = os.path.join(FILE_DIR, REPO, "processed", datatype)
    return pd.read_csv(os.path.join(path), sep='\t')


def _process_sentences(sentences: pd.DataFrame) -> npt.NDArray:
    return sentences['sentence'].unique()

def _process_chapters(chapters: pd.DataFrame) -> npt.NDArray:
    # XXX What to do about newlines?
    return chapters['tok'].to_numpy()

def _process_documents(chapters: pd.DataFrame) -> npt.NDArray:
    """
    So far: Articles, Magazines, Poems
    
    Strips markdown yaml from articles (when present)
    """
    # content_type: ['article' 'story' 'biblical text' 'blog post' 'chat' 'other' 'magazine' 'poem' 'screenplay' 'encyclopedia article']
    articles = chapters[chapters['content_type'] == 'article']['tok'].to_numpy()
    for i, a in enumerate(articles):
        yaml = "---"
        if a.startswith(yaml):
            st_idx = a.find(yaml, len(yaml))
            articles[i] = a[st_idx+len(yaml):].strip()

    magazines = chapters[chapters['content_type'] == 'magazine']['tok'].to_numpy()
    
    poems = chapters[chapters['content_type'] == 'poem']['tok'].to_numpy()

    bible = chapters[chapters['content_type'] == 'biblical text']
    # Filter out gospel of John
    # We have a bigger version elsewhere
    bible = bible.loc[bible["name"] != "gospel of john.txt"]
    bible = bible['tok'].to_numpy()

    story = chapters[chapters['content_type'] == 'story']

    # Remove more bible stuff
    story = story.drop(story.index[[0, 1, 2, 3]])
    story = story['tok'].to_numpy()

    blogpost = chapters[chapters['content_type'] == 'blog post']
    blogpost = blogpost['tok'].to_numpy()

    # Good amount of noise
    chat = chapters[chapters['content_type'] == 'chat']
    chat = chat['tok'].to_numpy()

    # encyclopedia seems to have extreme noise
    return np.concatenate((articles, magazines, poems, bible, story, blogpost, chat))


def load_data(datatype: StrEnum) -> npt.NDArray:
    data_path = _load_dataset(datatype)
    match datatype:
        case DataSets.chapters:
            return _process_chapters(data_path)
        case DataSets.documents:
            return _process_documents(data_path)
        case DataSets.sentences:
            return _process_sentences(data_path)
    raise ValueError(f"Invalid datatype: {datatype}")

def load_all_data() -> npt.NDArray:
    datas = []
    for data in DataSets:
        datas.append(load_data(data))

    return np.concatenate(datas)
