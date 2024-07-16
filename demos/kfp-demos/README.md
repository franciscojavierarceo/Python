# KFP Demos

Demo of various pipelines

Going to start small with

1. Pulling latest ticker data daily
2. Materializing it into SQLite
3. Transforming it and uploading it to online store
4. Training model automatically in a pipeline
5. Serving online
6. Web App to ask for should I buy?
 - This should call the online model to get a score and
    if the score > 0.7 return "no"
