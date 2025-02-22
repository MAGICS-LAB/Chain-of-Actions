from llama_index.core.llama_pack import download_llama_pack

# download_llama_pack("SearChainPack", "./searchain_pack")
from searchain_pack.llama_index.packs.searchain.base import SearChainPack
searchain = SearChainPack(
    data_path="data",
    dprtokenizer_path="./model/dpr_reader_multi",
    dprmodel_path="./model/dpr_reader_multi",
    crossencoder_name_or_path="./model/Quora_cross_encoder",
)
start_idx = 0
while not start_idx == -1:
    start_idx = execute(
        "/hotpotqa/hotpot_dev_fullwiki_v1_line.json", start_idx=start_idx
    )
