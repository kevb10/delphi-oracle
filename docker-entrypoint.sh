#!/bin/bash

# Set main app
echo "Set the main app"
export FLASK_APP=main.py

# Fix pytrends error
sed -i 's/pandas.io.json.normalize/pandas.io.json._normalize/g' /usr/local/lib/python3.7/site-packages/pytrends/request.py 

# Start server
echo "Starting server"
flask run --host=0.0.0.0

