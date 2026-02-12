# LLM Fine-Tuning for Information Extraction (NER)

This project provides a pipeline for fine-tuning the **Phi-3-mini-4k-instruct** model for **Information Extraction** tasks.

By leveraging **Unsloth** and **LoRA (Low-Rank Adaptation)**, we enable efficient fine-tuning on consumer-grade GPUs, transforming unstructured text into structured JSON data.

The primary goal of this model is to extract structured biographical information from unstructured narrative text. It acts as a specialized **Information Extraction** engine that parses natural language descriptions and outputs standardized JSON objects containing key entities.

**Capabilities:**

- **Entity Extraction:** Identifies and extracts specific entities such as `Name`, `Age`, `Job`, `Gender`.
- **Normalization:** Converts varied text descriptions (e.g., "twenty years old", "aged 76") into normalized structured formats.
- **Noise Reduction:** Filters out irrelevant narrative details to focus purely on the requested data schema.

### Example Training Data

The model is trained on pairs of unstructured prompts and their corresponding structured JSON responses:

> **Prompt:** "While strolling through a botanical garden, Igor, now 20 earns a living as a tour guide. He is known among friends for conducting amateur astronomy observations in quiet solitude."
>
> **Response:**
>
> ```json
> {
>   "name": "Igor",
>   "age": "20",
>   "job": "tour guide",
>   "gender": "male"
> }
> ```

## Features

- **Efficient Fine-Tuning:** Uses Unsloth's optimized kernels for 2x faster training and 60% less memory usage.
- **Configurable Pipeline:** All training parameters (learning rate, epochs, LoRA rank) are externalized in `config.yaml`.
- **GGUF Export:** Automatically exports the fine-tuned model to GGUF format for easy inference with tools like Ollama or llama.cpp.
- **Reproducible:** Python script and requirements file provided for consistent environment setup.

## Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd llm-fine-tuning-phi3
   ```

2. **Install Dependencies:**
   Ensure you have a CUDA-enabled GPU environment.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Configuration

Modify `config.yaml` to adjust model parameters, training settings, or data paths.

- **Model:** `unsloth/Phi-3-mini-4k-instruct-bnb-4bit`
- **Data:** `people_data.json`
- **Output:** `gguf_model_scratch_fixed`

### 2. Run Fine-Tuning

Execute the main script to start training and export the model:

```bash
python fine_tune.py
```

The script will:

1. Load the pre-trained Phi-3 model.
2. Apply LoRA adapters.
3. Train on the provided dataset.
4. Run a sample inference to verify performance.
5. Export the final model to GGUF format in the output directory.

### 3. Inference

You can run the model using Ollama (after creating a Modelfile) or llama.cpp.
The script already performs a basic inference sanity check at the end of training.

## Project Structure

- `fine_tune.py`: Main training script.
- `config.yaml`: Configuration file for all parameters.
- `requirements.txt`: Python dependencies.
- `fine_tune_LLM.ipynb`: Fine-tuning notebook, can be used on Google Colab.
- `people_data.json`: Dataset file.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improving the extraction accuracy or pipeline efficiency.
