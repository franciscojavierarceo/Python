# Graphql in Python

This is a light demo that shows graphql in a containerized flask application that allows you to interact with the query UI on http://127.0.0.1:5001/graphql by simply executing:

```
docker build .
docker-compose up
```
In `queries.graphql` you will see several valid queries that you can use to interact with the database / api.

Note, I use port 5001 because I had issue with 5000. 🤷‍♂️

After you run the commands above you'll be able to see the UI when you go to http://127.0.0.1:5001/graphql and see:

![GraphQL Image](/static/example-graphql.png)