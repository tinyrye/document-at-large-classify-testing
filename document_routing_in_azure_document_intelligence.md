# Question

what does the document routing look like in Azure Document Intelligence?

# Claude's Response

## Document Routing in Azure Document Intelligence

Azure Document Intelligence (formerly Form Recognizer) provides a **document classification and routing** capability that automatically identifies document types and routes them to the appropriate extraction model.

## Core Concept

Document routing uses a **custom classification model** as a front door that identifies the document type, then dispatches it to the correct extraction model:

```
Incoming Document (PDF/Image)
         │
         ▼
┌─────────────────────────┐
│  Custom Classifier Model │  ← "What type of document is this?"
│  (Document Routing)      │
└──────────┬──────────────┘
           │
     ┌─────┼──────┬──────────┐
     ▼     ▼      ▼          ▼
  Invoice  W-2  Receipt  Custom Form
  Model   Model  Model    Model
     │     │      │          │
     ▼     ▼      ▼          ▼
  Structured Extracted Fields/Values
```

## Custom Classification Model

This is the routing layer — you train it with labeled samples of each document type:

```python
from azure.ai.formrecognizer import DocumentModelAdministrationClient
from azure.core.credentials import AzureKeyCredential

endpoint = "https://<your-resource>.cognitiveservices.azure.com/"
key = "<your-key>"

admin_client = DocumentModelAdministrationClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

## Training data lives in Azure Blob Storage, organized by doc type:
## container/
##   ├── invoices/
##   │   ├── invoice1.pdf
##   │   ├── invoice2.pdf
##   ├── receipts/
##   │   ├── receipt1.pdf
##   │   ├── receipt2.pdf
##   ├── contracts/
##   │   ├── contract1.pdf

## Build the classifier
poller = admin_client.begin_build_document_classifier(
    doc_types={
        "invoice": ClassifierDocumentTypeDetails(
            azure_blob_source=AzureBlobContentSource(
                container_url="https://<storage>.blob.core.windows.net/training",
                prefix="invoices/"
            )
        ),
        "receipt": ClassifierDocumentTypeDetails(
            azure_blob_source=AzureBlobContentSource(
                container_url="https://<storage>.blob.core.windows.net/training",
                prefix="receipts/"
            )
        ),
        "contract": ClassifierDocumentTypeDetails(
            azure_blob_source=AzureBlobContentSource(
                container_url="https://<storage>.blob.core.windows.net/training",
                prefix="contracts/"
            )
        ),
    },
    description="Document router for incoming documents"
)

classifier = poller.result()
print(f"Classifier ID: {classifier.classifier_id}")
```

## Classify and Route at Runtime

```python
from azure.ai.formrecognizer import DocumentAnalysisClient

analysis_client = DocumentAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

## Step 1: Classify the document
with open("unknown_document.pdf", "rb") as f:
    poller = analysis_client.begin_classify_document(
        classifier_id="my-classifier-id",
        document=f
    )

classification_result = poller.result()

## Step 2: Route to the correct extraction model
MODEL_ROUTING = {
    "invoice":  "prebuilt-invoice",
    "receipt":  "prebuilt-receipt",
    "contract": "my-custom-contract-model",
    "w2":       "prebuilt-tax.us.w2",
}

for document in classification_result.documents:
    doc_type = document.doc_type
    confidence = document.confidence
    page_ranges = document.bounding_regions

    print(f"Detected: {doc_type} (confidence: {confidence:.2%})")
    print(f"Pages: {[r.page_number for r in page_ranges]}")

    # Route to the appropriate extraction model
    target_model = MODEL_ROUTING.get(doc_type)

    if target_model and confidence > 0.80:
        with open("unknown_document.pdf", "rb") as f:
            extract_poller = analysis_client.begin_analyze_document(
                model_id=target_model,
                document=f
            )
        extraction_result = extract_poller.result()

        for kv_pair in extraction_result.key_value_pairs:
            print(f"  {kv_pair.key.content}: {kv_pair.value.content}")
```

## Handling Multi-Document Files (Splitting)

A key feature — the classifier handles PDFs that contain **multiple document types** in a single file:

```python
## A single PDF might contain: pages 1-3 = invoice, pages 4-5 = receipt
with open("multi_doc_bundle.pdf", "rb") as f:
    poller = analysis_client.begin_classify_document(
        classifier_id="my-classifier-id",
        document=f
    )

result = poller.result()

## The classifier identifies EACH document within the file
for document in result.documents:
    print(f"Type: {document.doc_type}")
    print(f"Confidence: {document.confidence:.2%}")

    # Which pages belong to this logical document
    for region in document.bounding_regions:
        print(f"  Page: {region.page_number}")

## Output:
## Type: invoice      Confidence: 96.50%   Pages: 1, 2, 3
## Type: receipt       Confidence: 93.20%   Pages: 4, 5
```

```
multi_doc_bundle.pdf (5 pages)
  │
  ▼ Classifier
  │
  ├── Pages 1-3 → "invoice"  (96.5%) → prebuilt-invoice model
  └── Pages 4-5 → "receipt"  (93.2%) → prebuilt-receipt model
```

## Full Routing Pipeline

```python
class DocumentRouter:
    def __init__(self, endpoint, key, classifier_id):
        self.client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        self.classifier_id = classifier_id

        # Map document types to extraction models
        self.routing_table = {
            "invoice":       {"model": "prebuilt-invoice",     "min_confidence": 0.80},
            "receipt":       {"model": "prebuilt-receipt",     "min_confidence": 0.80},
            "w2":            {"model": "prebuilt-tax.us.w2",   "min_confidence": 0.85},
            "contract":      {"model": "custom-contract-v2",   "min_confidence": 0.75},
            "product_spec":  {"model": "custom-product-spec",  "min_confidence": 0.75},
        }

    def classify_and_extract(self, document_path: str) -> list[dict]:
        """Classify document(s) and route to appropriate extraction models."""

        # Step 1: Classify
        with open(document_path, "rb") as f:
            classify_poller = self.client.begin_classify_document(
                classifier_id=self.classifier_id,
                document=f
            )
        classification = classify_poller.result()

        results = []

        # Step 2: Route each detected document
        for doc in classification.documents:
            route = self.routing_table.get(doc.doc_type)

            if not route:
                results.append({
                    "type": doc.doc_type,
                    "status": "no_route_configured",
                    "confidence": doc.confidence
                })
                continue

            if doc.confidence < route["min_confidence"]:
                results.append({
                    "type": doc.doc_type,
                    "status": "below_confidence_threshold",
                    "confidence": doc.confidence,
                    "threshold": route["min_confidence"]
                })
                continue

            # Step 3: Extract with the routed model
            pages = [r.page_number for r in doc.bounding_regions]

            with open(document_path, "rb") as f:
                extract_poller = self.client.begin_analyze_document(
                    model_id=route["model"],
                    document=f,
                    pages=pages  # Only analyze relevant pages
                )
            extraction = extract_poller.result()

            results.append({
                "type": doc.doc_type,
                "status": "extracted",
                "confidence": doc.confidence,
                "pages": pages,
                "model_used": route["model"],
                "fields": {
                    field_name: field_value.content
                    for field_name, field_value in
                    (extraction.documents[0].fields or {}).items()
                    if field_value and field_value.content
                }
            })

        return results


## Usage
router = DocumentRouter(
    endpoint="https://my-resource.cognitiveservices.azure.com/",
    key="<key>",
    classifier_id="my-doc-classifier"
)

results = router.classify_and_extract("incoming_bundle.pdf")
for r in results:
    print(f"{r['type']} ({r['status']}): {r.get('fields', {})}")
```

## Available Prebuilt Models for Routing Targets

| Model ID | Document Type |
|-----------|--------------|
| `prebuilt-invoice` | Invoices |
| `prebuilt-receipt` | Receipts |
| `prebuilt-tax.us.w2` | W-2 tax forms |
| `prebuilt-tax.us.1099` | 1099 forms |
| `prebuilt-idDocument` | IDs, passports, driver's licenses |
| `prebuilt-healthInsuranceCard.us` | Insurance cards |
| `prebuilt-contract` | Contracts |
| `prebuilt-layout` | General layout extraction (tables, paragraphs, structure) |
| `prebuilt-read` | OCR / text extraction only |
| Custom models | Your fine-tuned extraction models |

## How It Connects to Your Existing Pattern

Given your config key `processing.classify.product-suggestions.llm-chat-prompting.enabled`, the routing concept maps naturally:

```properties
## Document Intelligence routing config
processing.classify.document-routing.enabled=true
processing.classify.document-routing.classifier-id=my-doc-classifier
processing.classify.document-routing.confidence-threshold=0.80

## Route-specific extraction models
processing.classify.document-routing.routes.invoice.model-id=prebuilt-invoice
processing.classify.document-routing.routes.product-spec.model-id=custom-product-spec-v2
processing.classify.document-routing.routes.contract.model-id=custom-contract-v3
```

> **Key takeaway:** Azure Document Intelligence's routing is a **two-stage pipeline** — classify first, then extract. The classifier learns document types from as few as **5 samples per type**, and the split-document capability makes it especially powerful for processing bulk scanned files containing mixed document types.