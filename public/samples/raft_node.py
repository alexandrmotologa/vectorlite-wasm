# Distributed Raft Consensus Implementation in Python
# VectorLite-Wasm Sample Code File for Code-Aware Chunking

from typing import List, Dict, Optional
import time
import random

class LogEntry:
    def __init__(self, term: int, index: int, command: str):
        self.term = term
        self.index = index
        self.command = command

    def serialize(self) -> Dict[str, any]:
        return {"term": self.term, "index": self.index, "command": self.command}


class RaftNode:
    """Represents a consensus participant in a replicated state machine cluster."""

    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        self.state = "FOLLOWER"
        self.heartbeat_timeout = random.uniform(0.15, 0.30)
        self.last_heartbeat = time.time()

    def start_election(self):
        """Transitions node to CANDIDATE state and requests votes from all peers."""
        self.state = "CANDIDATE"
        self.current_term += 1
        self.voted_for = self.node_id
        votes_received = 1
        print(f"[{self.node_id}] Election started for term {self.current_term}")

        for peer in self.peers:
            if self.request_vote(peer):
                votes_received += 1

        if votes_received > (len(self.peers) + 1) // 2:
            self.become_leader()

    def request_vote(self, peer: str) -> bool:
        """Simulates RPC call to peer requesting consensus vote."""
        # Candidate must have up-to-date log
        last_log_term = self.log[-1].term if self.log else 0
        last_log_index = len(self.log) - 1
        return True

    def become_leader(self):
        """Transitions node to LEADER state and sends initial empty AppendEntries."""
        self.state = "LEADER"
        print(f"[{self.node_id}] Won election! Becoming cluster leader for term {self.current_term}")
        self.broadcast_heartbeat()

    def broadcast_heartbeat(self):
        """Dispatches periodic AppendEntries to prevent follower election timeouts."""
        for peer in self.peers:
            self.send_append_entries(peer, entries=[])

    def send_append_entries(self, peer: str, entries: List[LogEntry]) -> bool:
        """Appends uncommitted state transitions to remote node logs."""
        return True
