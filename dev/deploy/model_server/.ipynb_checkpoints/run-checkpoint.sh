conda activate DeepRuleFastAPI

nohup uvicorn model_serve:app --reload --host 0.0.0.0 --port 6006 &