import requests

r=requests.get("https://rest.coinapi.io/v1/ohlcv/BITMEX_SPOT_BTC_USD/latest?period_id=1MIN", headers={"X-CoinAPI-Key":"0081D8E4-EA35-4ABE-8417-876F349564A5"})

print(r)
print(len(r.text))