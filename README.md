# 🏛️ Stoic LLM Steering

Steering language models to write like ancient Stoic philosophers using **Contrastive Activation Addition (CAA)** and **LoRA fine-tuning**.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)

---

## 📖 Overview

This project explores two techniques for steering a language model (Llama-3.2-1B) to generate text in the style of ancient Stoic philosophers:

1. **Contrastive Activation Addition (CAA)** - Extracts activation differences between Stoic and neutral text to create steering vectors
2. **LoRA Fine-tuning** - Parameter-efficient fine-tuning directly on classical Stoic texts

The project includes a complete data pipeline, model training, and an interactive Streamlit demo for comparing both approaches.

---

## ✨ Features

- 📚 **Automated data collection** from Project Gutenberg
- 🤖 **Neutral text generation** using Claude API
- 🎯 **CAA steering implementation** with adjustable coefficients
- 🔧 **LoRA fine-tuning** for three philosophers (Marcus Aurelius, Seneca, Epictetus)
- 🎨 **Interactive Streamlit demo** with side-by-side comparison
- 📊 **Systematic experimentation** across layers and hyperparameters

---

## 🚀 Quick Start

### Prerequisites
```bash
python 3.11+
conda or venv
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SebastianVillalobosAlva/Stoic-LLM-Steering.git
cd Stoic-LLM-Steering
```

2. **Create and activate environment**
```bash
conda create -n stoic-llm python=3.11
conda activate stoic-llm
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API key**
```bash
# Create .env file
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Demo
```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` to interact with the demo!

---

## 🏗️ Project Structure
```
Stoic-LLM-Steering/
├── packages/
│   ├── text_downloader/       # Project Gutenberg scraping
│   ├── paraphraser/           # Claude API neutral text generation
│   ├── steering_extractor/    # CAA steering vector extraction
│   ├── steering_runner/       # CAA inference
│   └── lora_trainer/          # LoRA fine-tuning
├── scripts/                   # Executable scripts
├── data/
│   ├── raw_texts/            # Downloaded Gutenberg texts
│   ├── processed/            # Stoic-neutral pairs
│   ├── steering_vectors/     # CAA vectors
│   └── lora_training/        # LoRA training data
├── lora_models/              # Trained LoRA adapters
├── streamlit_app.py          # Interactive demo
└── README.md
```

---

## 📊 Technical Approach

### 1. Data Pipeline

**Text Collection**
- Scraped classical Stoic texts from Project Gutenberg
- Philosophers: Marcus Aurelius, Seneca, Epictetus
- Filtered bibliographic content and religious passages

**Neutral Paraphrasing**
- Used Claude API to generate neutral versions of Stoic texts
- Created 30 contrastive pairs per philosopher
- Maintained semantic content while removing stylistic markers

### 2. Contrastive Activation Addition (CAA)

**Methodology**
```python
# Extract activations from last layer
stoic_activations = model(stoic_text)[-1]
neutral_activations = model(neutral_text)[-1]

steering_vector = mean(stoic_activations - neutral_activations) # Compute steering vector
output = mlp_output + coefficient * steering_vector # Apply at layer 12 during generation
```

**Optimal Settings**
- **Layer:** 12 (balanced between style and coherence)
- **Coefficient:** 0.08-0.11 (stronger = more influence, less coherence)

### 3. LoRA Fine-tuning

**Configuration**
- **Rank (r):** 8
- **Alpha:** 32
- **Target modules:** q_proj, v_proj
- **Trainable parameters:** 851,968 (0.07% of model)
- **Training epochs:** 3

**Results**
- More stable generation than CAA
- Better preservation of philosophical concepts
- Less sensitive to prompt phrasing

---

## 📈 Key Findings

### CAA vs LoRA Comparison

| Aspect | CAA | LoRA |
|--------|-----|------|
| **Training Time** | None (zero-shot) | ~15 min per philosopher |
| **Stability** | Moderate (prompt-sensitive) | High |
| **Style Transfer** | Weak-moderate | Strong |
| **Flexibility** | Adjustable at inference | Fixed after training |
| **Best Use Case** | Quick experiments | Production deployment |

### Challenges

1. **Small base model (1B params)** struggles with style preservation
2. **Religious content overlap** - Classical texts contain theological language that bleeds into steering
3. **Prompt sensitivity** - Declarative prompts work better than questions
4. **Data quality** - Only 30 pairs per philosopher limits steering strength

---

## 🎯 Results

### CAA Steering (Epictetus, coefficient=0.11)

**Prompt:** "When facing difficulty, one should"

**Output:**
> "When facing difficulty, one should not be afraid to admit it to oneself and others. In my own case, I am always a little ashamed of my weakness, and have a tendency to hide my weaknesses by doing a great deal of work. However, I am afraid that I have to be more honest about my weaknesses, lest I become a slave to my own weakness..."

### LoRA Fine-tuning (Marcus Aurelius)

**Prompt:** "To live well means"

**Output:**
> "To live well means to live according to nature, to follow reason, and to accept what cannot be changed. The wise man understands that external events are neither good nor bad, but his judgments make them so..."

---

## 🛠️ Reproducibility

### Generate Neutral Pairs
```bash
python -m scripts.text_downloader.main
```

### Extract CAA Steering Vectors
```bash
python -m scripts.steering_extractor.main
```

### Train LoRA Models
```bash
python -m scripts.lora_trainer.main
```

### Test Steering
```bash
python -m scripts.steering_runner.main
```

---

## 📚 Technologies Used

- **Transformers** - Model architecture and training
- **PEFT** - LoRA implementation
- **PyTorch** - Deep learning framework
- **Anthropic Claude API** - Neutral text generation
- **Streamlit** - Interactive demo
- **BeautifulSoup** - Web scraping

---

## 🎓 Learnings & Future Work

### Key Takeaways

1. **Model size matters** - Larger models (7B+) would likely show stronger steering effects
2. **Data quality > quantity** - Clean, diverse training pairs are crucial
3. **LoRA is more reliable** - For production use cases, fine-tuning beats zero-shot steering
4. **Activation engineering is powerful** - CAA works with zero training, just needs better base models

### Future Improvements

- [ ] Test on larger models (Llama-3.2-7B, Mistral-7B)
- [ ] Expand training data (100+ pairs per philosopher)
- [ ] Implement DPO/RLHF for preference learning
- [ ] Add more philosophers (Zeno, Chrysippus)
- [ ] Create evaluation metrics for "Stoic-ness"
- [ ] Deploy as public web app


## 👤 Author

**Sebastian Villalobos**
- GitHub: [@SebastianVillalobosAlva](https://github.com/SebastianVillalobosAlva)
- LinkedIn: [Linkedin](https://www.linkedin.com/in/sebastian-villalobos-alva/])

---

## 📧 Contact

Questions or interested in discussing this project? Feel free to reach out!

Built during Winter Break 2025 as an exploration of LLM steering techniques 🏛️

---

*"You have power over your mind - not outside events. Realize this, and you will find strength."* - Marcus Aurelius