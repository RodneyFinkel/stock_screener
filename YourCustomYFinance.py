import yfinance as yf
import sys
yf.enable_debug_mode()

print('run')
print(sys.version)

s_ticker = 'CSCO'
ticker = yf.Ticker(s_ticker)
# data = ticker.get_fast_info()
data = ticker.info
print(data)