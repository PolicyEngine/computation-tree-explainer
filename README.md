# computation-tree-explainer
Prototype Streamlit app to explain a computation tree with LLMs

Explanations use Claude Sonnet 5.5. The Anthropic SDK requires Python 3.10 or newer.

Install dependencies and run the offline explanation tests:

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

The tests use a mock transport and require no API credentials or provider calls.
