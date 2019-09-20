#!/bin/bash

# Set main app
export FLASK_APP=main.py

#sed -i 's/.normalize/._normalize/g'  '/usr/local/lib/python3.7/site-packages/pytrends/request.py'

# Start server
echo "Starting server"
flask run 
