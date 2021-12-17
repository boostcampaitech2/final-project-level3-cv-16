docker run -d \
    -p 8000:8000 \
    -v `pwd`/cache_path:/model_server/cache_path \
    --name model \
    model-0.1.0
