import os
import re

import fasttext
import pandas as pd
import polib
from huggingface_hub import hf_hub_download

PATH_TO_REPO = <SET ME>

model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)


def _parse(path_to_folder):
    data = []

    for root, _, files in os.walk(path_to_folder):
        for file in [f for f in files if f.endswith('.po')]:
            for entry in polib.pofile(os.path.join(root, file)):
                en = _clear(entry.msgid)
                tt = _clear(entry.msgstr)
                if en and tt and _check_is_tatar(tt):
                    data.append({
                        "en": en,
                        "tt": tt,
                        "src": f"xfce/{file}"
                    })

    df = pd.DataFrame(data)
    df.to_parquet("tt-en.parquet")


def _check_is_tatar(text) -> bool:
    """
    Check if the text is in Tatar language
    :param text
    :return: True if the text is in Tatar language, False otherwise
    """
    prediction = model.predict(text.replace("\n", " "))
    return prediction[0][0] == "__label__tat_Cyrl"


def _clear(text):
    # Remove html tags
    removed_tags = re.sub(r'<[^>]*>', '', text)
    # Remove underscores
    removed_underscores = re.sub(r'_', '', removed_tags)
    return removed_underscores.strip()


if __name__ == '__main__':
    _parse(PATH_TO_REPO)
