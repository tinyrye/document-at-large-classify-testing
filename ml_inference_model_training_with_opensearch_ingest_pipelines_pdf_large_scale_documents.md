
# Question

If i have an open search cluster, what options are there for ML inference to train a model based on PDFS with large paragraphs and extract specific information from trained locations based on phrasing and location?

**Selected Code**

```
processing.classify.product-suggestions.llm-chat-prompting.enabled
```

# Claude's Response



## ML Inference with OpenSearch for PDF Extraction

Given you already have an OpenSearch cluster, you have several strong options for training/deploying models that extract structured information from PDFs based on phrasing and positional context.

## Option 1: OpenSearch ML Framework (Native)

OpenSearch has a built-in ML Commons plugin that supports local model deployment:

```json
// Register a model in OpenSearch
POST /_plugins/_ml/models/_register
{
  "name": "my-document-extraction-model",
  "version": "1.0",
  "model_format": "TORCH_SCRIPT",
  "model_config": {
    "model_type": "bert",
    "embedding_dimension": 768,
    "framework_type": "sentence_transformers"
  },
  "url": "https://your-internal-artifact-store/model.zip"
}
```

**What it gives you:**
- Neural search pipelines
- Semantic search over ingested PDF chunks
- Ingest processors that call ML models at index time
- k-NN vector search for similarity matching

### Ingest Pipeline with ML Inference

```json
PUT _ingest/pipeline/pdf-extraction-pipeline
{
  "description": "Extract and embed PDF content",
  "processors": [
    {
      "text_embedding": {
        "model_id": "<your-deployed-model-id>",
        "field_map": {
          "passage_text": "passage_embedding"
        }
      }
    }
  ]
}
```

## Option 2: OpenSearch + Layout-Aware Document AI Models

This is the best fit for your use case — extracting information from **specific locations** in PDFs based on **phrasing and spatial position**.

### Architecture

```
PDFs
  │
  ▼
┌─────────────────────────┐
│  Document Parsing Layer  │
│  (PDF → structured data) │
│  - Apache Tika           │
│  - PyMuPDF / pdfplumber  │
│  - Unstructured.io       │
│  - Amazon Textract (*)   │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Layout-Aware ML Model   │
│  (understands position)  │
│  - LayoutLMv3            │
│  - Donut                 │
│  - DocTR                 │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  OpenSearch Cluster      │
│  - Store extracted fields│
│  - Vector embeddings     │
│  - Semantic search       │
└─────────────────────────┘
```

### Key Models for Position-Aware Extraction

| Model | Strength | Local? |
|-------|----------|--------|
| **LayoutLMv3** (Microsoft) | Understands text + layout + visual features together. Fine-tunable for your specific PDF formats. | ✅ |
| **Donut** (Naver) | OCR-free document understanding — directly reads document images | ✅ |
| **DocTR** (Mindee) | OCR + layout detection, open source | ✅ |
| **Unstructured.io** | Parses PDFs into structured elements (titles, paragraphs, tables) with position metadata | ✅ |
| **LlamaIndex** | Orchestrates extraction + indexing into OpenSearch | ✅ |

### LayoutLMv3 Fine-Tuning Example (Best for Your Use Case)

Since you want to extract from **trained locations based on phrasing and position**:

```python
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from datasets import Dataset
import torch

## LayoutLMv3 takes: tokens + bounding boxes + image
## You fine-tune it to label specific regions/fields

processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", 
    apply_ocr=True  # built-in OCR
)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(your_label_set)  # e.g., PRODUCT_NAME, PRICE, DESCRIPTION, etc.
)

## Your training data: PDFs annotated with bounding boxes + labels
## Tools for annotation: Label Studio, Prodigy, or custom
training_data = [
    {
        "image": pdf_page_image,
        "tokens": ["Product", "Name:", "Widget-X"],
        "bboxes": [[72, 100, 200, 120], [72, 100, 250, 120], [260, 100, 400, 120]],
        "labels": ["B-FIELD_LABEL", "I-FIELD_LABEL", "B-FIELD_VALUE"]
    }
]

## Fine-tune with HuggingFace Trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./layoutlm-finetuned",
    num_train_epochs=30,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## Option 3: OpenSearch + Unstructured.io + RAG Pipeline

For a quicker path without fine-tuning a layout model:

```python
from unstructured.partition.pdf import partition_pdf
from opensearchpy import OpenSearch

## 1. Parse PDFs with layout awareness
elements = partition_pdf(
    filename="your_document.pdf",
    strategy="hi_res",           # Uses detectron2 for layout detection
    infer_table_structure=True,  # Extracts tables
    extract_images_in_pdf=True,
)

## Each element has metadata including:
## - coordinates (bounding box)
## - element type (Title, NarrativeText, Table, ListItem, etc.)
## - page number

for element in elements:
    print(f"Type: {type(element).__name__}")
    print(f"Text: {element.text}")
    print(f"Coordinates: {element.metadata.coordinates}")
    print(f"Page: {element.metadata.page_number}")

## 2. Index into OpenSearch with positional metadata
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    use_ssl=True,
)

for i, element in enumerate(elements):
    doc = {
        "text": element.text,
        "element_type": type(element).__name__,
        "page_number": element.metadata.page_number,
        "coordinates": str(element.metadata.coordinates),
        "source_file": "your_document.pdf",
        # Add vector embedding for semantic search
        "text_embedding": get_embedding(element.text),  # from local model
    }
    client.index(index="pdf-extractions", id=i, body=doc)
```

## Option 4: OpenSearch Neural Search + SageMaker (If AWS)

If your OpenSearch is on AWS (Amazon OpenSearch Service):

```json
// Create an ML connector to a SageMaker endpoint running LayoutLMv3
POST /_plugins/_ml/connectors/_create
{
  "name": "LayoutLM Extraction Connector",
  "description": "Connect to SageMaker-hosted LayoutLMv3",
  "version": 1,
  "protocol": "aws_sigv4",
  "parameters": {
    "region": "us-east-1",
    "service_name": "sagemaker"
  },
  "actions": [
    {
      "action_type": "predict",
      "method": "POST",
      "url": "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/layoutlm-endpoint/invocations",
      "headers": {
        "content-type": "application/json"
      },
      "request_body": "{ \"inputs\": \"${parameters.input}\" }"
    }
  ]
}
```

## Recommended Approach for Your Use Case

Given your requirements (**PDFs with large paragraphs**, **extract from trained locations**, **phrasing and position**):

```
Best Path:
──────────

1. PDF Parsing:       Unstructured.io (hi_res strategy)
                      → preserves bounding boxes, element types, page structure

2. Extraction Model:  LayoutLMv3 fine-tuned on YOUR PDFs
                      → learns "field X is always at position Y with phrasing Z"
                      → needs ~50-200 annotated pages to get good results

3. Annotation Tool:   Label Studio (open source, self-hosted)
                      → annotate PDFs with bounding boxes + field labels

4. Storage/Search:    OpenSearch with k-NN + neural search
                      → index extracted fields + embeddings
                      → semantic queries over extracted content

5. Orchestration:     Your Spring Boot app calls the pipeline
                      → similar pattern to your existing chatService.getAnswer()
```

```properties
## Your existing config pattern extends naturally:
processing.classify.product-suggestions.document-extraction.enabled=true
processing.classify.product-suggestions.document-extraction.model-endpoint=http://localhost:8501/v1/extract
processing.classify.product-suggestions.document-extraction.opensearch-index=pdf-extractions
```

> **Key insight:** LayoutLMv3 is the strongest option here because it jointly models **text content + spatial position + visual features** — which directly maps to your need for "trained locations based on phrasing and location." Standard NLP models (BERT, GPT, etc.) throw away positional information from the PDF layout, which is exactly what you need to preserve.