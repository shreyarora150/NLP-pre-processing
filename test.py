import pytest
import pathlib
import platform
import nbimporter
import pickle as pkl
import pandas as pd
from pandas.testing import assert_frame_equal
from spacy.tokens.doc import Doc
from spacy.lang.en import English
from pandas import DataFrame
from dataclasses import dataclass

from preprocessing import (extract_text, clean_text,
                           process_text, to_dataframe,
                           customize_tokenizer)


@dataclass
class Shared:
    nlp: English
    noisy_target_text: str
    cleaned_target_text: str
    target_doc: Doc
    target_df: DataFrame

   
@pytest.fixture(scope="session")
def shared():
    ext = ""
    if platform.system() == "Windows":
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        ext = "_win"
    with open(f"test_utils/nlp{ext}.pkl", "rb") as pklfile:
       nlp = pkl.load(pklfile)
    with open("test_utils/text.txt", encoding="utf8") as tfile:
        noisy_target_text = tfile.read()
    with open("test_utils/cleaned_text.txt", encoding="utf8") as tfile:
        cleaned_target_text = tfile.read()
    with open(f"test_utils/doc{ext}.pkl", "rb") as pklfile:
       target_doc = pkl.load(pklfile)
    target_df = pd.read_csv("test_utils/df.csv", encoding="utf8").fillna('')
    return Shared(nlp, noisy_target_text, cleaned_target_text, target_doc, target_df)

   
def test_extract_text(shared):
    with open("world-food-prices.html", encoding="utf8") as html_file:
        html_content = html_file.read()
    test_text = extract_text(html_content)
    assert shared.noisy_target_text == test_text[:580]
   
   
def test_clean_text(shared):
    cleaned_test_text = clean_text(shared.noisy_target_text)
    assert shared.cleaned_target_text == cleaned_test_text


def test_process_text(shared):
    test_doc = process_text(shared.cleaned_target_text, shared.nlp)
    assert Doc == type(test_doc)
    assert len(shared.target_doc) == len(test_doc)
    target_tokens = [token.text for token in shared.target_doc]
    test_tokens = [token.text for token in test_doc]
    assert target_tokens == test_tokens
    target_ents = [ent.text for ent in shared.target_doc.ents]
    test_ents = [ent.text for ent in test_doc.ents]
    assert target_ents == test_ents

   
def test_to_dataframe(shared):
    test_df = to_dataframe(shared.target_doc)
    assert_frame_equal(shared.target_df, test_df, check_dtype=False)


def test_customize_tokenizer(shared):
    target_tokens = ['06', '/', '01', '/', '2023']
    test_customized_nlp = customize_tokenizer(shared.nlp)
    test_doc = test_customized_nlp("06/01/2023")
    test_tokens = [token.text for token in test_doc]
    assert target_tokens == test_tokens
