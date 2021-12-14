docker run -i \
    -p 8000:8000 \
    -v `pwd`/cache_path:/model_server/cache_path \
    --name model_ \
    model-0.1.0
