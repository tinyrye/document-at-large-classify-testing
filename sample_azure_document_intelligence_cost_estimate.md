
[Estimate Script](estimate_azure_document_intelligence.py)

Route: invoices
  Documents/month:   10,000
  Pages/month:       30,000
  Classification:    $300.00
  Extraction:        $300.00
  Route Total:       $600.00
  Per Document:      $0.06

Route: receipts
  Documents/month:   25,000
  Pages/month:       25,000
  Classification:    $250.00
  Extraction:        $250.00
  Route Total:       $500.00
  Per Document:      $0.02

Route: contracts
  Documents/month:   2,000
  Pages/month:       24,000
  Classification:    $240.00
  Extraction:        $1,200.00   ← custom model = 5x prebuilt
  Query Fields:      $120.00
  Route Total:       $1,560.00
  Per Document:      $0.78

Route: product_specs
  Documents/month:   3,000
  Pages/month:       24,000
  Classification:    $240.00
  Extraction:        $1,200.00
  Query Fields:      $120.00
  Route Total:       $1,560.00
  Per Document:      $0.52

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL MONTHLY:       $4,220.00
TOTAL PAGES:         103,000
AVG COST/PAGE:       $0.04097