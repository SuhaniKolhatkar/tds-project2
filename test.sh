#!/bin/bash
echo "Testing health endpoint..."
curl http://localhost:5000/health

echo -e "\n\nTesting quiz endpoint with demo..."
curl -X POST http://localhost:5000/quiz \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$EMAIL\",
    \"secret\": \"$SECRET\",
    \"url\": \"https://tds-llm-analysis.s-anand.net/demo\"
  }"