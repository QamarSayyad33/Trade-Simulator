This project develops a high-performance trade simulator leveraging real-time Level 2 
orderbook data from the OKX cryptocurrency exchange. The simulator processes streaming 
market data, estimates transaction costs including slippage, fees, and market impact, and 
displays these through an interactive user interface optimized for low latency and accuracy. 
2. Problem Statement 
Cryptocurrency trading incurs costs and risks due to slippage, fees, and market impact, which 
traders must estimate to make informed decisions. This assignment requires building a system 
that consumes OKX’s real-time L2 orderbook WebSocket feed and applies quantitative models 
to dynamically estimate these costs while maintaining efficient processing and UI 
responsiveness. 
3. Tools and Technologies 
• Programming Language: Python 
• WebSocket: websockets with asyncio for asynchronous data streaming 
• UI Framework: Streamlit 
• Data Processing: Pandas, NumPy 
• Modelling: scikit-learn, statsmodels 
• Logging: Python logging module 
• Data Formats: JSON parsing 
• Visualization: Matplotlib, Plotly
