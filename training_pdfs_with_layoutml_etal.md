# Question

for Open Search or Elaticsearch ML Pipelines, are there examples on github of people training a model against a large document format where the text is in an index?

# Claude's Response

## ML Pipelines with OpenSearch/Elasticsearch for Training on Indexed Documents

Yes, there are several examples and approaches. Let me break down what's available:

## OpenSearch ML Commons Plugin

The [OpenSearch ML Commons plugin](https://github.com/opensearch-project/ml-commons) is the primary repository for ML functionality in OpenSearch. It supports:

- Training models on data stored in OpenSearch indices
- Deploying models for inference within ingest/search pipelines
- Integration with external models (SageMaker, Bedrock, etc.)

```json
// Example: Training a model against an index
POST /_plugins/_ml/_train/kmeans
{
  "parameters": {
    "centroids": 3,
    "iterations": 10
  },
  "input_query": {
    "query": { "match_all": {} },
    "_source": ["text_field"],
    "size": 10000
  },
  "input_index": ["my-documents-index"]
}
```

## Key GitHub Repositories & Examples

### 1. **OpenSearch Neural Search**
- [opensearch-project/neural-search](https://github.com/opensearch-project/neural-search)
- Uses ingest pipelines to generate embeddings from text in indices

### 2. **OpenSearch ML Commons Demos**
- [opensearch-project/ml-commons/docs](https://github.com/opensearch-project/ml-commons/tree/main/docs)
- Contains tutorials on training and deploying models

### 3. **Semantic Search with OpenSearch**
```json
// Ingest pipeline that generates embeddings from indexed text
PUT _ingest/pipeline/nlp-pipeline
{
  "description": "Generate embeddings for text",
  "processors": [
    {
      "text_embedding": {
        "model_id": "<model_id>",
        "field_map": {
          "document_text": "document_embedding"
        }
      }
    }
  ]
}
```

### 4. **Elasticsearch Examples**
- [elastic/elasticsearch-labs](https://github.com/elastic/elasticsearch-labs) — Notebooks and examples for ML with Elasticsearch
- [elastic/eland](https://github.com/elastic/eland) — Python library for ML with Elasticsearch, including importing models and running inference

## Common Patterns People Use

### Pattern A: Train Externally, Deploy into OpenSearch
```python
## Using eland to upload a trained model to Elasticsearch
from eland.ml.pytorch import PyTorchModel

PyTorchModel.import_model(
    es_client=es,
    model_id="my-text-model",
    model=traced_model,
    tokenizer=tokenizer,
    task_type="text_embedding"
)
```

### Pattern B: Use Bulk Data from Index for Fine-Tuning
```python
## Scroll through an index to extract training data
from opensearchpy import OpenSearch, helpers

docs = helpers.scan(
    client,
    index="my-large-documents",
    query={"query": {"match_all": {}}},
    _source=["text_field", "label"]
)

training_data = [(doc["_source"]["text_field"], doc["_source"]["label"]) for doc in docs]
## Fine-tune your model with this data
```

### Pattern C: Reindex Through an ML Pipeline
```json
// Reindex existing documents through an ML ingest pipeline
POST _reindex
{
  "source": { "index": "raw-documents" },
  "dest": {
    "index": "enriched-documents",
    "pipeline": "nlp-pipeline"
  }
}
```

## Notable Community Examples

| Repository | Description |
|---|---|
| [opensearch-project/ml-commons](https://github.com/opensearch-project/ml-commons) | Core ML plugin with training APIs |
| [elastic/elasticsearch-labs](https://github.com/elastic/elasticsearch-labs) | Jupyter notebooks for ES ML workflows |
| [opensearch-project/opensearch-py-ml](https://github.com/opensearch-project/opensearch-py-ml) | Python ML client for OpenSearch (similar to eland) |
| [aryn-ai/sycamore](https://github.com/aryn-ai/sycamore) | Document processing pipelines that integrate with OpenSearch |

## Practical Advice

If your goal is to **train/fine-tune a model on large documents stored in an index**, the most common real-world approach is:

1. **Extract** text from the index using scroll/scan API or `opensearch-py-ml`/`eland`
2. **Train/fine-tune** externally (PyTorch, Hugging Face, etc.)
3. **Upload** the model back into OpenSearch/Elasticsearch
4. **Create an ingest pipeline** using the deployed model
5. **Reindex** your documents through the pipeline

The built-in `_train` API in OpenSearch ML Commons is more limited (k-means, anomaly detection, etc.) and not suited for training large language or embedding models directly on-cluster.

---

# Question

what examples are there of people searching on opensearch or elasticsearch using the neural search plugin?

**Selected Code**

```
https://github.com/opensearch-project/neural-search
```

# Claude's Response



## Neural Search Examples with OpenSearch

There are quite a few examples across GitHub, blog posts, and official documentation. Here's a comprehensive roundup:

## Official OpenSearch Neural Search Examples

### 1. **OpenSearch Documentation Samples**
The [opensearch-project/documentation-website](https://github.com/opensearch-project/documentation-website) contains detailed walkthroughs.

### Basic Neural Search Setup (End-to-End)

**Step 1: Register and deploy a model**
```json
// Register a pretrained model
POST /_plugins/_ml/models/_register
{
  "name": "huggingface/sentence-transformers/all-MiniLM-L12-v2",
  "version": "1.0.1",
  "model_group_id": "<group_id>",
  "model_format": "TORCH_SCRIPT"
}

// Deploy the model
POST /_plugins/_ml/models/<model_id>/_deploy
```

**Step 2: Create an ingest pipeline**
```json
PUT /_ingest/pipeline/neural-search-pipeline
{
  "description": "Pipeline for generating embeddings",
  "processors": [
    {
      "text_embedding": {
        "model_id": "<model_id>",
        "field_map": {
          "text": "text_embedding"
        }
      }
    }
  ]
}
```

**Step 3: Create an index with knn vector field**
```json
PUT /my-neural-index
{
  "settings": {
    "index.knn": true,
    "default_pipeline": "neural-search-pipeline"
  },
  "mappings": {
    "properties": {
      "text": { "type": "text" },
      "text_embedding": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "engine": "lucene",
          "space_type": "l2",
          "name": "hnsw",
          "parameters": {}
        }
      }
    }
  }
}
```

**Step 4: Ingest documents**
```json
POST /my-neural-index/_doc
{
  "text": "OpenSearch is a community-driven open source search and analytics suite."
}
```

**Step 5: Neural search query**
```json
GET /my-neural-index/_search
{
  "query": {
    "neural": {
      "text_embedding": {
        "query_text": "open source search engine",
        "model_id": "<model_id>",
        "k": 10
      }
    }
  }
}
```

## GitHub Repositories with Working Examples

### 2. **opensearch-project/opensearch-py-ml**
[GitHub Link](https://github.com/opensearch-project/opensearch-py-ml)

```python
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False
)

## Neural search query in Python
query = {
    "query": {
        "neural": {
            "text_embedding": {
                "query_text": "What is machine learning?",
                "model_id": "your-model-id",
                "k": 5
            }
        }
    }
}

response = client.search(index="my-neural-index", body=query)
for hit in response["hits"]["hits"]:
    print(f"Score: {hit['_score']}, Text: {hit['_source']['text']}")
```

### 3. **Hybrid Search (Neural + BM25)**
This is one of the most popular patterns:

```json
GET /my-neural-index/_search
{
  "query": {
    "hybrid": {
      "queries": [
        {
          "match": {
            "text": {
              "query": "open source search engine"
            }
          }
        },
        {
          "neural": {
            "text_embedding": {
              "query_text": "open source search engine",
              "model_id": "<model_id>",
              "k": 10
            }
          }
        }
      ]
    }
  },
  "search_pipeline": {
    "phase_results_processors": [
      {
        "normalization-processor": {
          "normalization": { "technique": "min_max" },
          "combination": {
            "technique": "arithmetic_mean",
            "parameters": { "weights": [0.3, 0.7] }
          }
        }
      }
    ]
  }
}
```

### 4. **Multimodal Neural Search**
```json
GET /my-multimodal-index/_search
{
  "query": {
    "neural": {
      "image_embedding": {
        "query_text": "a photo of a cat sitting on a couch",
        "model_id": "<multimodal_model_id>",
        "k": 5
      }
    }
  }
}
```

### 5. **Neural Sparse Search (Sparse Encoding)**
```json
// Ingest pipeline for sparse encoding
PUT /_ingest/pipeline/sparse-pipeline
{
  "processors": [
    {
      "sparse_encoding": {
        "model_id": "<sparse_model_id>",
        "field_map": {
          "text": "text_sparse"
        }
      }
    }
  ]
}

// Sparse search query
GET /my-sparse-index/_search
{
  "query": {
    "neural_sparse": {
      "text_sparse": {
        "query_text": "open source analytics",
        "model_id": "<sparse_model_id>"
      }
    }
  }
}
```

## Community / Third-Party Examples

| Repository / Resource | Description |
|---|---|
| [dylanjcastillo/opensearch-neural-search-tutorial](https://github.com/dylanjcastillo/opensearch-neural-search-tutorial) | Step-by-step Python tutorial |
| [opensearch-project/opensearch-dashboards-notebooks](https://github.com/opensearch-project/dashboards-notebooks) | Interactive notebook examples |
| [aryn-ai/sycamore](https://github.com/aryn-ai/sycamore) | Document ETL into OpenSearch with neural search |
| [zilliztech/VectorDBBench](https://github.com/zilliztech/VectorDBBench) | Benchmarking vector search including OpenSearch |
| [aws-samples/opensearch-genai-samples](https://github.com/aws-samples) | AWS samples using neural search with Bedrock |
| [flower-org/flower-search](https://github.com/adithya-s-k/AI-Engineering-Cohort) | RAG implementations using OpenSearch neural search |

### 6. **RAG (Retrieval-Augmented Generation) with Neural Search**

This is the most popular community use case:

```python
from opensearchpy import OpenSearch
import openai

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

## Step 1: Neural search for relevant context
search_response = client.search(
    index="knowledge-base",
    body={
        "query": {
            "neural": {
                "text_embedding": {
                    "query_text": "How does OpenSearch handle vector search?",
                    "model_id": "<model_id>",
                    "k": 5
                }
            }
        },
        "_source": ["text", "title"]
    }
)

## Step 2: Build context from results
context = "\n\n".join([
    hit["_source"]["text"] 
    for hit in search_response["hits"]["hits"]
])

## Step 3: Send to LLM
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": f"Answer based on this context:\n{context}"},
        {"role": "user", "content": "How does OpenSearch handle vector search?"}
    ]
)
```

### 7. **Conversational Search (OpenSearch 2.12+)**
```json
// Using search pipeline with RAG processor
PUT /_search/pipeline/rag-pipeline
{
  "response_processors": [
    {
      "retrieval_augmented_generation": {
        "model_id": "<llm_model_id>",
        "context_field_list": ["text"],
        "system_prompt": "You are a helpful assistant.",
        "user_instructions": "Answer the question based on the provided context."
      }
    }
  ]
}

// Query using the RAG pipeline
GET /my-neural-index/_search?search_pipeline=rag-pipeline
{
  "query": {
    "neural": {
      "text_embedding": {
        "query_text": "What are the benefits of neural search?",
        "model_id": "<embedding_model_id>",
        "k": 5
      }
    }
  },
  "ext": {
    "generative_qa_parameters": {
      "llm_question": "What are the benefits of neural search?"
    }
  }
}
```

## Docker Compose for Local Testing

Many examples include a docker-compose setup:

```yaml
version: '3'
services:
  opensearch:
    image: opensearchproject/opensearch:2.13.0
    environment:
      - discovery.type=single-node
      - plugins.ml_commons.only_run_on_ml_node=false
      - plugins.ml_commons.model_access_control_enabled=false
      - plugins.ml_commons.native_memory_threshold=99
      - OPENSEARCH_INITIAL_ADMIN_PASSWORD=MyStr0ng!Pass
    ports:
      - "9200:9200"
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536
    mem_limit: 4g

  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:2.13.0
    ports:
      - "5601:5601"
    environment:
      OPENSEARCH_HOSTS: '["https://opensearch:9200"]'
```

## Best Starting Points

If you want to get hands-on quickly, I'd recommend:

1. **Quickest start**: [dylanjcastillo's tutorial](https://github.com/dylanjcastillo/opensearch-neural-search-tutorial) — simple Python + Docker
2. **Official reference**: [OpenSearch Neural Search docs](https://opensearch.org/docs/latest/search-plugins/neural-search/)
3. **Production patterns**: The hybrid search approach (BM25 + neural) is what most production deployments use
4. **RAG applications**: The `aws-samples` repos on GitHub have several end-to-end RAG examples using OpenSearch as the vector store

---

# Question

What examples are there of people using PyTorch to train ML Models on large PDFs or Word Docs to classify sections/paragraphs as containing key metrics extractable information entities?

**Selected Code**

```
https://github.com/opensearch-project/documentation-website
```

# Claude's Response



## PyTorch Models for Document Section Classification & Entity Extraction from PDFs/Word Docs

This is a rich area spanning document AI, information extraction, and layout-aware NLP. Here's a comprehensive survey of what's available:

## Foundation Models for Document Understanding

### 1. **LayoutLM / LayoutLMv2 / LayoutLMv3 (Microsoft)**
The most widely used family for document section classification and entity extraction.

| Repository | Description |
|---|---|
| [microsoft/unilm/layoutlm](https://github.com/microsoft/unilm/tree/master/layoutlm) | Original LayoutLM |
| [microsoft/unilm/layoutlmv2](https://github.com/microsoft/unilm/tree/master/layoutlmv2) | Adds visual features |
| [microsoft/unilm/layoutlmv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) | Unified text + layout + image |
| [huggingface/transformers](https://github.com/huggingface/transformers) | All LayoutLM variants integrated |

```python
import torch
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3TokenizerFast
)
from PIL import Image
from pdf2image import convert_from_path

## Convert PDF page to image
images = convert_from_path("financial_report.pdf")
image = images[0]

## Initialize processor and model
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=10  # e.g., revenue, date, metric_name, metric_value, etc.
)

## Process document
encoding = processor(
    image,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

## Inference
with torch.no_grad():
    outputs = model(**encoding)
    predictions = torch.argmax(outputs.logits, dim=-1)
```

### 2. **Donut (Document Understanding Transformer)**
[GitHub: clovaai/donut](https://github.com/clovaai/donut)

OCR-free document understanding — processes document images directly.

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

image = Image.open("report_page.png")

## Ask a question about the document
task_prompt = "<s_docvqa><s_question>What is the total revenue?</s_question><s_answer>"
decoder_input_ids = processor.tokenizer(
    task_prompt, add_special_tokens=False, return_tensors="pt"
).input_ids

pixel_values = processor(image, return_tensors="pt").pixel_values

outputs = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
)

answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

### 3. **DocFormer**
[GitHub: shabie/docformer](https://github.com/shabie/docformer)

Multi-modal transformer combining text, vision, and spatial features.

## Full Training Pipeline Examples

### 4. **Training LayoutLM for Custom Entity Extraction on Documents**

This is the most common pattern people use:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    AdamW,
    get_scheduler
)
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import json

## ============================================================
## Step 1: Define custom labels for your domain
## ============================================================
LABEL_LIST = [
    "O",                    # Outside any entity
    "B-METRIC_NAME",        # Beginning of a metric name (e.g., "Revenue")
    "I-METRIC_NAME",        # Inside a metric name
    "B-METRIC_VALUE",       # Beginning of a metric value (e.g., "$1.2B")
    "I-METRIC_VALUE",       # Inside a metric value
    "B-DATE",               # Beginning of a date
    "I-DATE",
    "B-SECTION_HEADER",     # Section titles
    "I-SECTION_HEADER",
    "B-KPI",                # Key performance indicator
    "I-KPI",
    "B-ENTITY_NAME",        # Company, product, person
    "I-ENTITY_NAME",
    "B-PERCENTAGE",
    "I-PERCENTAGE",
    "B-CURRENCY_AMOUNT",
    "I-CURRENCY_AMOUNT",
]

label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}

## ============================================================
## Step 2: Document Dataset
## ============================================================
class DocumentEntityDataset(Dataset):
    """
    Expects annotations in a format like:
    {
        "file": "report.pdf",
        "page": 0,
        "annotations": [
            {
                "text": "Revenue",
                "bbox": [100, 200, 250, 220],
                "label": "B-METRIC_NAME"
            },
            {
                "text": "$1.2B",
                "bbox": [300, 200, 400, 220],
                "label": "B-METRIC_VALUE"
            }
        ]
    }
    """
    def __init__(self, annotation_file, processor, max_length=512):
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]

        # Convert PDF page to image
        images = convert_from_path(
            item["file"],
            first_page=item["page"] + 1,
            last_page=item["page"] + 1
        )
        image = images[0]

        # Extract words and bounding boxes via OCR
        ocr_result = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )

        words = []
        boxes = []
        word_labels = []

        width, height = image.size

        for i in range(len(ocr_result["text"])):
            word = ocr_result["text"][i].strip()
            if not word:
                continue

            # Normalize bbox to 0-1000 range (LayoutLM convention)
            x0 = int(1000 * ocr_result["left"][i] / width)
            y0 = int(1000 * ocr_result["top"][i] / height)
            x1 = int(1000 * (ocr_result["left"][i] + ocr_result["width"][i]) / width)
            y1 = int(1000 * (ocr_result["top"][i] + ocr_result["height"][i]) / height)

            words.append(word)
            boxes.append([x0, y0, x1, y1])

            # Match to annotation labels
            label = self._match_label(word, [x0, y0, x1, y1], item["annotations"])
            word_labels.append(label2id[label])

        # Process through LayoutLMv3
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Squeeze batch dimension
        return {k: v.squeeze(0) for k, v in encoding.items()}

    def _match_label(self, word, bbox, annotations):
        """Match a word to its annotation label based on bbox overlap."""
        for ann in annotations:
            ann_bbox = self._normalize_bbox(ann["bbox"])
            if self._iou(bbox, ann_bbox) > 0.5:
                return ann["label"]
        return "O"

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def _normalize_bbox(self, bbox):
        return bbox  # Assume already normalized


## ============================================================
## Step 3: Training Loop
## ============================================================
def train():
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False  # We handle OCR ourselves
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    dataset = DocumentEntityDataset("annotations.json", processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 10
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    model.save_pretrained("./trained_document_model")
    processor.save_pretrained("./trained_document_model")

if __name__ == "__main__":
    train()
```

### 5. **Inference Pipeline: Extract Metrics from New Documents**

```python
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from pdf2image import convert_from_path
import pytesseract

class DocumentMetricExtractor:
    def __init__(self, model_path):
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_path, apply_ocr=False
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def extract_from_pdf(self, pdf_path):
        """Extract all metrics from all pages of a PDF."""
        images = convert_from_path(pdf_path)
        all_entities = []

        for page_num, image in enumerate(images):
            entities = self._process_page(image, page_num)
            all_entities.extend(entities)

        return self._group_entities(all_entities)

    def _process_page(self, image, page_num):
        # OCR
        ocr = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words, boxes = [], []
        width, height = image.size

        for i in range(len(ocr["text"])):
            word = ocr["text"][i].strip()
            if not word:
                continue
            x0 = int(1000 * ocr["left"][i] / width)
            y0 = int(1000 * ocr["top"][i] / height)
            x1 = int(1000 * (ocr["left"][i] + ocr["width"][i]) / width)
            y1 = int(1000 * (ocr["top"][i] + ocr["height"][i]) / height)
            words.append(word)
            boxes.append([x0, y0, x1, y1])

        encoding = self.processor(
            image, words, boxes=boxes,
            return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]

        # Map predictions back to words
        entities = []
        word_ids = encoding.word_ids(0)
        previous_word_id = None

        for idx, (pred, word_id) in enumerate(zip(predictions, word_ids)):
            if word_id is None or word_id == previous_word_id:
                continue
            label = self.id2label[pred.item()]
            if label != "O":
                entities.append({
                    "word": words[word_id],
                    "label": label,
                    "bbox": boxes[word_id],
                    "page": page_num,
                    "confidence": torch.softmax(outputs.logits[0][idx], dim=-1).max().item()
                })
            previous_word_id = word_id

        return entities

    def _group_entities(self, entities):
        """Group B- and I- tagged tokens into complete entities."""
        grouped = []
        current_entity = None

        for ent in entities:
            if ent["label"].startswith("B-"):
                if current_entity:
                    grouped.append(current_entity)
                current_entity = {
                    "type": ent["label"][2:],
                    "text": ent["word"],
                    "page": ent["page"],
                    "confidence": ent["confidence"]
                }
            elif ent["label"].startswith("I-") and current_entity:
                if ent["label"][2:] == current_entity["type"]:
                    current_entity["text"] += " " + ent["word"]
                    current_entity["confidence"] = min(
                        current_entity["confidence"], ent["confidence"]
                    )

        if current_entity:
            grouped.append(current_entity)

        return grouped


## Usage
extractor = DocumentMetricExtractor("./trained_document_model")
results = extractor.extract_from_pdf("quarterly_report.pdf")

for entity in results:
    print(f"[{entity['type']}] {entity['text']} "
          f"(page {entity['page']}, confidence: {entity['confidence']:.2f})")

## Output:
## [METRIC_NAME] Total Revenue (page 2, confidence: 0.95)
## [METRIC_VALUE] $4.2 billion (page 2, confidence: 0.93)
## [DATE] Q3 2024 (page 2, confidence: 0.97)
## [PERCENTAGE] 12.5% (page 3, confidence: 0.91)
## [KPI] Year-over-year growth (page 3, confidence: 0.88)
```

## Specialized Repositories & Projects

### 6. **Document Section Classification**

[GitHub: allenai/VILA](https://github.com/allenai/VILA) — Visual Layout-aware pre-training for scientific documents.

```python
## Section classification approach using BERT + spatial features
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SectionClassifier(nn.Module):
    """
    Classifies document paragraphs into sections like:
    - Executive Summary
    - Financial Metrics
    - Risk Factors
    - Key Performance Indicators
    - Methodology
    """
    def __init__(self, num_sections=10):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 64),  # bbox features
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_sections)
        )

    def forward(self, input_ids, attention_mask, bbox_features):
        text_features = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output  # [batch, 768]

        spatial_features = self.spatial_encoder(bbox_features)  # [batch, 128]

        combined = torch.cat([text_features, spatial_features], dim=-1)
        logits = self.classifier(combined)
        return logits
```

### 7. **LILT (Language-Independent Layout Transformer)**
[GitHub: jpWang/LiLT](https://github.com/jpWang/LiLT)

Works across languages, great for multilingual document extraction.

### 8. **Impira / DocQuery**
[GitHub: impira/docquery](https://github.com/impira/docquery)

```python
from docquery import document, pipeline

p = pipeline("document-question-answering")
doc = document.load_document("financial_report.pdf")

## Ask questions to extract metrics
questions = [
    "What is the total revenue?",
    "What is the net income?",
    "What is the year-over-year growth rate?",
    "What is the operating margin?",
]

for q in questions:
    result = p(question=q, **doc.context)
    print(f"Q: {q}")
    print(f"A: {result['answer']} (confidence: {result['score']:.2f})")
```

### 9. **Deepdoctection**
[GitHub: deepdoctection/deepdoctection](https://github.com/deepdoctection/deepdoctection)

Full document analysis pipeline with layout detection, table recognition, and text extraction.

```python
import deepdoctection as dd

analyzer = dd.get_dd_analyzer()

## Process a PDF
path = "annual_report.pdf"
df = analyzer.analyze(path=path)

for page in df:
    for layout in page.layouts:
        print(f"Type: {layout.category_name}")
        print(f"Text: {layout.text}")
        print(f"Bbox: {layout.bounding_box}")
        print("---")
```

## Word Document Processing

### 10. **Processing .docx Files for Training**

```python
from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

class DocxFeatureExtractor:
    """Extract paragraphs with structural metadata from Word docs."""

    SECTION_LABELS = {
        "executive_summary": 0,
        "financial_metrics": 1,
        "risk_factors": 2,
        "kpi_section": 3,
        "methodology": 4,
        "appendix": 5,
        "table_of_contents": 6,
        "narrative": 7,
    }

    def extract_paragraphs(self, docx_path):
        doc = Document(docx_path)
        paragraphs = []

        for i, para in enumerate(doc.paragraphs):
            if not para.text.strip():
                continue

            features = {
                "text": para.text,
                "style": para.style.name if para.style else "Normal",
                "is_heading": para.style.name.startswith("Heading") if para.style else False,
                "heading_level": self._get_heading_level(para),
                "is_bold": any(run.bold for run in para.runs if run.bold is not None),
                "is_italic": any(run.italic for run in para.runs if run.italic is not None),
                "font_sizes": list(set(
                    run.font.size.pt for run in para.runs
                    if run.font.size is not None
                )),
                "alignment": str(para.alignment) if para.alignment else "LEFT",
                "word_count": len(para.text.split()),
                "has_numbers": any(c.isdigit() for c in para.text),
                "has_currency": any(sym in para.text for sym in ["$", "€", "£", "¥"]),
                "has_percentage": "%" in para.text,
                "position_ratio": i / max(len(doc.paragraphs), 1),
                "paragraph_index": i,
            }
            paragraphs.append(features)

        return paragraphs

    def _get_heading_level(self, para):
        if para.style and para.style.name.startswith("Heading"):
            try:
                return int(para.style.name.split()[-1])
            except ValueError:
                return 0
        return 0

    def extract_tables(self, docx_path):
        """Extract tables which often contain key metrics."""
        doc = Document(docx_path)
        tables_data = []

        for table_idx, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(cells)

            tables_data.append({
                "table_index": table_idx,
                "rows": rows,
                "num_rows": len(rows),
                "num_cols": len(rows[0]) if rows else 0,
            })

        return tables_data


## Training dataset from Word docs
class WordDocClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_path, tokenizer, max_length=256):
        """
        annotations_path: JSON file with format:
        [
            {
                "docx_path": "report.docx",
                "paragraph_index": 5,
                "label": "financial_metrics"
            },
            ...
        ]
        """
        with open(annotations_path) as f:
            self.annotations = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extractor = DocxFeatureExtractor()
        self.label_map = DocxFeatureExtractor.SECTION_LABELS

        # Pre-extract all paragraphs
        self._cache = {}
        for ann in self.annotations:
            path = ann["docx_path"]
            if path not in self._cache:
                self._cache[path] = self.extractor.extract_paragraphs(path)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        paragraphs = self._cache[ann["docx_path"]]
        para = paragraphs[ann["paragraph_index"]]

        # Encode text
        encoding = self.tokenizer(
            para["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Structural features
        structural = torch.tensor([
            para["heading_level"] / 6.0,
            float(para["is_bold"]),
            float(para["is_italic"]),
            float(para["has_numbers"]),
            float(para["has_currency"]),
            float(para["has_percentage"]),
            para["position_ratio"],
            para["word_count"] / 500.0,
        ], dtype=torch.float32)

        label = self.label_map[ann["label"]]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "structural_features": structural,
            "labels": torch.tensor(label, dtype=torch.long)
        }
```

## Annotation Tools People Use

To create training data for these models:

| Tool | Link | Description |
|---|---|---|
| **Label Studio** | [github.com/HumanSignal/label-studio](https://github.com/HumanSignal/label-studio) | Open source, supports PDF/document annotation |
| **UBIAI** | [ubiai.tools](https://ubiai.tools) | Specialized for document entity annotation |
| **Prodigy** | [prodi.gy](https://prodi.gy) | By Explosion (spaCy), fast annotation |
| **Doccano** | [github.com/doccano/doccano](https://github.com/doccano/doccano) | Open source text annotation |
| **INCEpTION** | [github.com/inception-project/inception](https://github.com/inception-project/inception) | Semantic annotation platform |

## End-to-End GitHub Projects

| Repository | Stars | Description |
|---|---|---|
| [microsoft/unilm](https://github.com/microsoft/unilm) | 19k+ | LayoutLM family, state-of-the-art |
| [deepdoctection/deepdoctection](https://github.com/deepdoctection/deepdoctection) | 2k+ | Full pipeline: layout + OCR + extraction |
| [clovaai/donut](https://github.com/clovaai/donut) | 5k+ | OCR-free document understanding |
| [NielsRogge/Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials) | 8k+ | Excellent LayoutLM fine-tuning notebooks |
| [katanaml/sparrow](https://github.com/katanaml/sparrow) | 1k+ | Data extraction from documents with ML |
| [applicaai/kleister-nda](https://github.com/applicaai/kleister-nda) | — | NDA document key info extraction benchmark |
| [doc-analysis/DocBank](https://github.com/doc-analysis/DocBank) | — | 500K document pages with token-level annotations |
| [ibm-aur-nlp/PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) | — | 1M+ document images for layout analysis |
| [jpWang/LiLT](https://github.com/jpWang/LiLT) | 1k+ | Language-independent layout transformer |

## Recommended Starting Path

For your specific use case of "classifying sections/paragraphs as containing key metrics and extractable entities":

1. **Start with pre-trained LayoutLMv3** — it understands document layout natively
2. **Use Label Studio** to annotate 200-500 document pages with your custom labels
3. **Fine-tune using the HuggingFace Trainer** — see [NielsRogge's tutorials](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LayoutLMv3)
4. **For Word docs specifically**, extract to images or use the text+structural features approach above
5. **Deploy into OpenSearch** using an ingest pipeline for at-scale processing

```python
## Quick-start: Fine-tune LayoutLMv3 with HuggingFace Trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
```