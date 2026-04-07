#!/usr/bin/env python

class DocumentIntelligenceCostEstimator:
    """Estimate Azure Document Intelligence costs based on routing topology."""

    # Per-page pricing (USD) — verify against current Azure pricing page
    PRICING = {
        "custom-classification":    0.01,
        "prebuilt-read":            0.001,
        "prebuilt-layout":          0.01,
        "prebuilt-invoice":         0.01,
        "prebuilt-receipt":         0.01,
        "prebuilt-tax.us.w2":       0.01,
        "prebuilt-idDocument":      0.01,
        "prebuilt-contract":        0.01,
        "custom-extraction":        0.05,
        "addon-query-fields":       0.005,  # per page, added on top
    }

    def __init__(self, routes: dict):
        """
        routes: {
            "doc_type": {
                "model": "prebuilt-invoice" | "custom-extraction",
                "avg_pages_per_doc": 3,
                "monthly_volume": 5000,
                "uses_query_fields": False,
            }
        }
        """
        self.routes = routes

    def estimate_monthly_cost(self) -> dict:
        breakdown = {}
        total_cost = 0.0
        total_pages = 0

        for doc_type, config in self.routes.items():
            volume = config["monthly_volume"]
            avg_pages = config["avg_pages_per_doc"]
            total_pages_for_type = volume * avg_pages

            # Cost 1: Classification (every page goes through this)
            classification_cost = total_pages_for_type * self.PRICING["custom-classification"]

            # Cost 2: Extraction (routed model processes the pages)
            model = config["model"]
            extraction_price = self.PRICING.get(model, 0.05)  # default to custom
            extraction_cost = total_pages_for_type * extraction_price

            # Cost 3: Optional add-on query fields
            addon_cost = 0.0
            if config.get("uses_query_fields", False):
                addon_cost = total_pages_for_type * self.PRICING["addon-query-fields"]

            route_total = classification_cost + extraction_cost + addon_cost

            breakdown[doc_type] = {
                "monthly_documents": volume,
                "avg_pages_per_doc": avg_pages,
                "total_pages": total_pages_for_type,
                "classification_cost": round(classification_cost, 2),
                "extraction_cost": round(extraction_cost, 2),
                "addon_cost": round(addon_cost, 2),
                "route_total": round(route_total, 2),
                "cost_per_document": round(route_total / volume, 4),
            }

            total_cost += route_total
            total_pages += total_pages_for_type

        return {
            "routes": breakdown,
            "summary": {
                "total_monthly_cost": round(total_cost, 2),
                "total_monthly_pages": total_pages,
                "avg_cost_per_page": round(total_cost / total_pages, 5) if total_pages else 0,
            }
        }


## ─── Example Usage ───

estimator = DocumentIntelligenceCostEstimator(
    routes={
        "invoices": {
            "model": "prebuilt-invoice",
            "avg_pages_per_doc": 3,
            "monthly_volume": 10_000,
            "uses_query_fields": False,
        },
        "receipts": {
            "model": "prebuilt-receipt",
            "avg_pages_per_doc": 1,
            "monthly_volume": 25_000,
            "uses_query_fields": False,
        },
        "contracts": {
            "model": "custom-extraction",
            "avg_pages_per_doc": 12,
            "monthly_volume": 2_000,
            "uses_query_fields": True,
        },
        "product_specs": {
            "model": "custom-extraction",
            "avg_pages_per_doc": 8,
            "monthly_volume": 3_000,
            "uses_query_fields": True,
        },
    }
)

result = estimator.estimate_monthly_cost()