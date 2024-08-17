import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError

with open('database.json') as f:
    document = json.load(f)

with open('BERT_database.json') as f:
    schema = json.load(f)


try:
    validate(instance=document, schema=schema)
    print("Validation is complete without errors")
except ValidationError as e:
    print("Validation is failed")
    print(f"Errors: {e.massage}")
