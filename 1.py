from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize, get_tokenizer
import torch

model, preprocess = create_model_from_pretrained('conch_ViT-B-16',
                                                 "/home/ubuntu/zhonghaiqin/virtual_stain/UNSB/checkpoint/conch/pytorch_model.bin")

tokenizer = get_tokenizer()


def is_fully_encoded(text: str):
    # 关键：这里不要截断，先看真实长度
    out = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    n_tokens = len(out["input_ids"])
    fully_encoded = n_tokens <= 127  # 127 以内才会完整进入 CONCH
    return fully_encoded, n_tokens


text = "Breast pathology image, H&E to HER2 IHC. Preserve tumor architecture, lesion boundary, and cell distribution. Emphasize membranous HER2 expression, staining intensity, staining completeness, and positive region distribution. H&E shows blue-purple nuclei and pink-red stroma. HER2 IHC typically shows brown membranous staining in positive tumor cells with blue or light blue nuclear counterstain, and a pale white stromal background."

ok, n = is_fully_encoded(text)
print(f"真实token长度: {n}, 是否完整编码: {ok}")

# token_ids = tokenize(tokenizer, ["Breast pathology image, H&E to HER2 IHC. Preserve tumor architecture, lesion boundary, and cell distribution. Emphasize membranous HER2 expression, staining intensity, staining completeness, and positive region distribution. H&E shows blue-purple nuclei and pink-red stroma. HER2 IHC typically shows brown membranous staining in positive tumor cells with blue or light blue nuclear counterstain, and a pale white stromal background."])
