from electrock_infer.engine.llm_engine import LLMEngine

import warnings
# 忽略 Pytorch 的这几条特定弃用警告
warnings.filterwarnings('ignore', message=".*'has_cuda' is deprecated.*")
warnings.filterwarnings('ignore', message=".*'has_cudnn' is deprecated.*")
warnings.filterwarnings('ignore', message=".*'has_mps' is deprecated.*")
warnings.filterwarnings('ignore', message=".*'has_mkldnn' is deprecated.*")
class LLM(LLMEngine):
    pass
