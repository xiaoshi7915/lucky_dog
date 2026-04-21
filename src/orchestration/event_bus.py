"""轻量事件总线。"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any
from queue import Queue


EventHandler = Callable[[dict[str, Any]], None]


class EventBus:
    """发布订阅事件总线。"""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._queues: dict[str, Queue[dict[str, Any]]] = defaultdict(Queue)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """为事件注册处理函数。"""
        self._handlers[event_name].append(handler)

    def publish(self, event_name: str, payload: dict[str, Any]) -> None:
        """发布事件并广播给订阅者。"""
        self._queues[event_name].put(payload)
        for handler in self._handlers[event_name]:
            handler(payload)

    def poll(self, event_name: str) -> dict[str, Any] | None:
        """获取事件队列中的一条消息。"""
        queue = self._queues[event_name]
        if queue.empty():
            return None
        return queue.get()
