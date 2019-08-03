# DELPHI
Stock predictor

## Available scripts
### `python manage.py migrate`
Propagate changes made to models to the database schema.

### `python manage.py runserver`
Runs the app at [http://localhost:8000](http://localhost:8000)

## Endpoints
### Load chart
[http://delphi.foussa.io/api/v0.1/load/{TICKER_SYMBOL}](http://delphi.foussa.io/api/v0.1/)
Retrieves the historical performance of the stock.

### Evaluate chart
[http://delphi.foussa.io/api/v0.1/evaluate/{TICKER_SYMBOL}](http://delphi.foussa.io/api/v0.1/)
Retrieves the historical performance of the stock and the historical perdictions.
