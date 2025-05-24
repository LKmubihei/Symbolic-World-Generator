# 🧠 Symbolic World Generator (SWG-RL)

Welcome to the official repository of **Symbolic World Generator (SWG-RL)** — a domain-level symbolic planner powered by large language models .

This repository contains code and datasets for evaluating different language models' ability to generate PDDL domains from natural language descriptions.

## Project Structure

```
.
├── dataset/               # Test datasets
│   └── test/             # Test cases with descriptions and ground truth
├── scripts/              # Evaluation scripts
│   ├── test_model.py     # Main evaluation script
│   └── run_all.sh        # Script to run evaluation for all models
├── util/                 # Utility functions
└── results/              # Evaluation results
```




## Installation




1. Clone the repository:
```bash
git clone https://github.com/yourusername/pddl-domain-evaluation.git
cd pddl-domain-evaluation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install docker for running LAPKT planner. There are two different docker images available:
   - The docker image for the linux/arm64 platform is available [here](<https://hub.docker.com/repository/docker/gautierdag/lapkt-arm/general>). See the `Dockerfile` for more details.
   - The docker image for the linux/amd64 platform is available [here](<https://hub.docker.com/r/lapkt/lapkt-public>)


## Usage

1. Run evaluation for a single model:
```bash
python scripts/test_model.py <model_name>
```

2. Run evaluation for all models:
```bash
bash scripts/run_all.sh
```


## 🚀 Model Release

We are excited to announce that the **SWG-RL trained model** is now **open-sourced** on Hugging Face!

👉 **[Click here to explore the model](https://huggingface.co/lkmubihei/SWG-RL)**

The model is fine-tuned using curriculum learning and guided reinforcement over symbolic planning tasks. It can be used to generate valid and executable PDDL domains from natural language.

---


