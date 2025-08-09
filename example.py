import os
from electrock_infer import LLM, SamplingParams
from transformers import AutoTokenizer
from electrock_infer.flash_infer.engine_core import EngineCore

def main():
    # path = os.path.expanduser("/work/home/ac1m1kpqy8/zhusg/models/Qwen3-0.6B")
    # path = os.path.expanduser("/work/share/data/XDZS2025/Mixtral-8x7B-v0.1")
    path = os.path.expanduser("/work/home/ac1m1kpqy8/zhusg/models/AI-ModelScope/Mixtral-8x7B-v0_1/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    # if tokenizer.chat_template is None:
    #     chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
    #     tokenizer.chat_template = chat_template

    # llm = LLM(path, 
    #           enforce_eager=True, 
    #           tensor_parallel_size=2,
    #           gpu_memory_utilization=0.9)
    llm = EngineCore(path, 
              enforce_eager=True, 
              tensor_parallel_size=2,
              gpu_memory_utilization=0.9)


    sampling_params = SamplingParams(temperature=0.9, max_tokens=256)
    prompts = [
        "hello",
        "list all prime nmbers within 100",
        "Hello my name is"
    ]
    # prompts = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "user", "content": prompt}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #         enable_thinking=False
    #     )
    #     for prompt in prompts
    # ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
