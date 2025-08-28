# TableZoomer

<div align="center">

![TableZoomer Overview](Overview%20for%20TableZoomer.png)


</div>

## 📖 Overview

TableZoomer is a sophisticated multi-agent framework designed for table question answering (TQA) tasks. It employs a ReAct (Reasoning and Acting) approach with multiple specialized agents working collaboratively to understand, analyze, and answer questions about tabular data.


## 🚀 Quick Start

### Prerequisites
Our framework is built on the [MetaGPT](https://github.com/geekan/MetaGPT) library.
```angular2html
Python 3.9 or later, but less than 3.12
metagpt >= 0.8.1
openai == 1.6.1
vllm (optional)
```

### Basic Usage
### 1. Configure model interfaces and prompt templates.
- Create the config file in `agent_config/`. Example: `agent_config/example.yaml`
    
- Set the LLMs and prompt templates you prefer in the config file.
    -  LLM config (Refer to `agent_config/qwen3-8b_api.yaml`):

        For close-sourced api, usage:
        ```
        llm:
        api_type: "openai"  # or azure / ollama / groq etc.
        model: "gpt-4-turbo"  # or gpt-3.5-turbo
        base_url: "https://api.openai.com/v1"  # or forward url / other llm url
        api_key: "YOUR_API_KEY"
        ```

        For the model employed as described in [Model Deployment](#model_dep), usage:
        ```
        llm:
        api_type: open_llm
        base_url: "http://{ip}:{port}/v1"
        model: "{model}"
        temperature: 0
        calc_usage: false
        api_key: your_api_key
        ```

    - Prompt config
        Example prompts: `prompts/*.txt`

### 2. RUN
- Main program entrance: `table_agent.py`.

    Run table_agent.py with your table, query and agent config file.
    ```
    python table_agent.py
    ```

- Run on DataBench: `databench_infer.py`
    ```
    python databench_infer.py
    ```

