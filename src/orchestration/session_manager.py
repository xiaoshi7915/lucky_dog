"""会话管理模块：按 session_id 维护独立的 RealtimeLoop 实例。"""

from __future__ import annotations

import threading
from time import time
from typing import Any

from src.config import AppConfig


class SessionEntry:
    """单个会话的元数据与 RealtimeLoop 引用。"""

    def __init__(self, session_id: str, loop: Any, mode: str) -> None:
        self.session_id = session_id
        self.loop = loop  # RealtimeLoop 实例
        self.mode = mode
        self.created_at: float = time()
        self.last_active_at: float = time()
        self.turn_count: int = 0
        self.status: str = "active"

    def touch(self) -> None:
        """更新最后活跃时间。"""
        self.last_active_at = time()

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。"""
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "status": self.status,
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
        }


class SessionManager:
    """线程安全的会话注册与查询管理器。

    每个会话持有独立的 RealtimeLoop，避免多用户并发共享同一实例
    导致的状态竞争问题。
    """

    def __init__(self, config: AppConfig, ttl_sec: int = 1800, max_sessions: int = 50) -> None:
        self._config = config
        self._ttl_sec = ttl_sec  # 会话无活动超时秒数（默认 30 分钟）
        self._max_sessions = max_sessions
        self._store: dict[str, SessionEntry] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: str, mode: str = "") -> SessionEntry:
        """创建新会话并初始化独立的 RealtimeLoop。

        若 session_id 已存在则直接返回已有会话（幂等操作）。
        """
        # 延迟导入避免循环依赖
        from src.orchestration.realtime_loop import RealtimeLoop  # noqa: PLC0415

        with self._lock:
            self._evict_expired()
            if session_id in self._store:
                return self._store[session_id]
            if len(self._store) >= self._max_sessions:
                self._evict_oldest()
            loop = RealtimeLoop(config=self._config)
            entry = SessionEntry(session_id=session_id, loop=loop, mode=mode or self._config.runtime_mode)
            self._store[session_id] = entry
            return entry

    def get_session(self, session_id: str) -> SessionEntry | None:
        """查询会话，不存在或已过期则返回 None。"""
        with self._lock:
            self._evict_expired()
            return self._store.get(session_id)

    def end_session(self, session_id: str) -> bool:
        """结束并移除指定会话，返回是否成功。"""
        with self._lock:
            entry = self._store.pop(session_id, None)
            if entry:
                entry.status = "ended"
            return entry is not None

    def list_sessions(self) -> list[dict[str, Any]]:
        """返回所有活跃会话的摘要列表。"""
        with self._lock:
            self._evict_expired()
            return [entry.to_dict() for entry in self._store.values()]

    def _evict_expired(self) -> None:
        """清理超时的会话（调用方需持有锁）。"""
        now = time()
        expired = [sid for sid, entry in self._store.items() if now - entry.last_active_at > self._ttl_sec]
        for sid in expired:
            self._store.pop(sid, None)

    def _evict_oldest(self) -> None:
        """超出上限时淘汰最久未活跃的会话（调用方需持有锁）。"""
        if not self._store:
            return
        oldest_id = min(self._store, key=lambda sid: self._store[sid].last_active_at)
        self._store.pop(oldest_id, None)
