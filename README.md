# chainlit-chatbot

Cloning the project:
```bash
git clone https://github.com/MMPuyanfar/chainlit-chatbot.git
cd chainlit-chatbot
```
Creating a virtual environment and installing the dependencies:
(3.13 >= python version >= 3.12)
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/mac
pip install -r requirements.txt
```

Running the project (run the command in the root directory):
```bash
chainlit run app.py -w
```

If you want to run the tests, first install the test libraries:
```bash
pip install -r test-requirements.txt
```

Then run the tests:
```bash
pytest
```

To create a vector store, prepare the data in json format with 'question' and 'answer', then modify *FAQ_DATA_PATH* according to your data file (It should be a relateive path like 'data/your_data.json').
After that, run the qdrant docker image:
```bash
mkdir -p ~/qdrant_storage
docker run -d --name qdran -p 6333:6333 -p 6334:6334 -v ~/qdrant_storage:/qdrant/storage:Z qdrant/qdrant
```
Then use the ingest script to bulk load your data. From the project root, run:
```bash
python scripts/ingest.py
```
Alternatively, you can use the ingest API to add question/answer pairs one-by-one. In order to use the API, run:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```



Then run the chainlit app and use the RAG pipeline