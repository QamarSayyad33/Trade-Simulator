import asyncio
import time
import aiohttp
import orjson
from typing import Dict, Optional

class OKXClient:
    """
    WebSocket client to connect to OKX real-time L2 orderbook data for a specified asset.
    Uses asyncio and aiohttp for asynchronous WebSocket handling.
    Maintains an internal queue to process incoming messages off the main receive loop.
    Includes a watchdog to send pings and detect stalled connections.
    """

    def __init__(self, asset: str):
        """
        Initialize the client for a given trading asset.
        :param asset: Asset symbol, e.g., "BTC-USDT"
        """
        self.asset = asset

        # Stores the latest received and parsed data as a dictionary
        self.data: Dict = {}

        # aiohttp WebSocket connection object (None when disconnected)
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None

        # Running flag to control lifecycle of client tasks
        self.running = False

        # Timestamp of the last received message, used by watchdog
        self.last_msg_time = time.time()

        # Asyncio queue to hold raw incoming messages for separate processing
        self.message_queue: asyncio.Queue = asyncio.Queue()

        # Background asyncio tasks for watchdog and message processing
        self.watchdog_task: Optional[asyncio.Task] = None
        self.processor_task: Optional[asyncio.Task] = None

    async def _process_messages(self):
        """
        Background task that continuously consumes messages from the queue,
        deserializes JSON content, updates internal data, and tracks last message time.
        Uses a timeout to allow graceful exit when no messages arrive.
        """
        while self.running:
            try:
                # Wait up to 1 second for a message to arrive in queue
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No message received within timeout, loop again to check running flag
                continue

            try:
                # Deserialize JSON message using orjson (fast deserialization)
                self.data = orjson.loads(message)

                # Update last message time to current timestamp
                self.last_msg_time = time.time()
            except Exception as e:
                # Log but do not raise exceptions on deserialization errors
                print(f"[Processor] Failed to process message: {e}")
            finally:
                # Mark message as processed in queue
                self.message_queue.task_done()

    async def _watchdog(self):
        """
        Background watchdog task to periodically send ping messages to keep connection alive.
        If no message has been received in the last 25 seconds, it attempts to ping.
        If sending ping fails, it stops the client to trigger reconnect or shutdown.
        """
        while self.running:
            await asyncio.sleep(5)  # Check every 5 seconds

            # Check if last message time was more than 25 seconds ago
            if time.time() - self.last_msg_time > 25:
                try:
                    if self.ws:
                        await self.ws.send_str("ping")
                        # Optional: print("[Watchdog] Sent ping")
                except Exception as e:
                    print(f"[Watchdog] Error sending ping: {e}")
                    # Stop running on failure to send ping, to trigger reconnect or cleanup
                    self.running = False

    async def _connect_once(self):
        """
        Connects once to the WebSocket endpoint and manages receiving messages.
        Sets up background tasks for watchdog and message processing.
        Cleans up tasks on disconnect or errors.
        """
        # Construct WebSocket URL for given asset's L2 orderbook
        url = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{self.asset}-SWAP"

        # Create a new aiohttp ClientSession to manage HTTP/WebSocket connection lifecycle
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                print(f"[WebSocket] Connected to {url}")

                # Save WebSocket reference and mark client as running
                self.ws = ws
                self.running = True
                self.last_msg_time = time.time()

                # Start background watchdog and message processor tasks
                self.watchdog_task = asyncio.create_task(self._watchdog())
                self.processor_task = asyncio.create_task(self._process_messages())

                # Main loop to receive messages from the WebSocket
                async for msg in ws:
                    if not self.running:
                        # Exit loop if client is stopped externally
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # Enqueue text messages for async processing
                        await self.message_queue.put(msg.data)

                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        print("[WebSocket] Closed")
                        break

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"[WebSocket] Error: {msg.data}")
                        break

                # After loop ends, mark running as False to indicate disconnect
                self.running = False

                # Cancel and await background tasks to clean up
                if self.watchdog_task:
                    self.watchdog_task.cancel()
                    try:
                        await self.watchdog_task
                    except asyncio.CancelledError:
                        pass

                if self.processor_task:
                    self.processor_task.cancel()
                    try:
                        await self.processor_task
                    except asyncio.CancelledError:
                        pass

                # Clear WebSocket reference
                self.ws = None

    async def connect(self):
        """
        Public method to establish and maintain WebSocket connection.
        Reconnects on connection failures with incremental backoff.
        Exits when `self.running` is False.
        """
        retry = 0
        while True:
            try:
                await self._connect_once()
            except Exception as e:
                print(f"[WebSocket] Connection failed: {e}. Retrying in 5s...")
                retry += 1

                # Backoff delay up to max 30 seconds
                await asyncio.sleep(min(5 * retry, 30))

            # Stop reconnecting if running flag is cleared externally
            if not self.running:
                break

    def get_data(self):
        """
        Return the latest parsed data snapshot.
        :return: dict of latest orderbook data
        """
        return self.data

    def stop(self):
        """
        Stop the client by setting running flag to False.
        Background tasks and connection will exit on their own.
        """
        self.running = False

# Alias to match your existing code
WebsocketManager = OKXClient
