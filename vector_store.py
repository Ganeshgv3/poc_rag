"""Vector backends: local/remote ChromaDB (default) or Qdrant (VECTOR_BACKEND=qdrant)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import chromadb

# sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM = 384

STREAMLIT_REGISTRY_COLLECTION = "__streamlit_file_registry__"
DOC_PAYLOAD_KEY = "_text"


def _stable_point_id(s: str) -> int:
    digest = hashlib.sha256(s.encode("utf-8")).digest()[:8]
    val = int.from_bytes(digest, "big", signed=False)
    return val if val != 0 else 1


def _is_qdrant(client: Any) -> bool:
    return type(client).__module__.startswith("qdrant_client")


def create_vector_client(persist_directory: Path) -> Any:
    backend = (os.getenv("VECTOR_BACKEND") or "chroma").strip().lower()
    if backend == "qdrant":
        try:
            from qdrant_client import QdrantClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VECTOR_BACKEND=qdrant requires qdrant-client. Install: pip install qdrant-client"
            ) from exc

        url = (os.getenv("QDRANT_URL") or "http://localhost:6333").strip()
        api_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
        return QdrantClient(url=url, api_key=api_key)

    url = (os.getenv("CHROMA_URL") or os.getenv("CHROMA_SERVER_URL") or "").strip()
    if url:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 8000)
        ssl = parsed.scheme == "https"
        return chromadb.HttpClient(host=host, port=port, ssl=ssl)
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_directory))


def get_or_create_vector_collection(client: Any, collection_name: str) -> Any:
    if _is_qdrant(client):
        return QdrantCollectionAdapter(client, collection_name)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def delete_named_collection(client: Any, name: str) -> None:
    if _is_qdrant(client):
        if client.collection_exists(collection_name=name):
            client.delete_collection(collection_name=name)
        return
    client.delete_collection(name=name)


def _record_from_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "filename": meta["filename"],
        "stored_name": meta["stored_name"],
        "path": meta["path"],
        "sha256": meta["sha256"],
        "collection_name": meta["collection_name"],
        "chunks": int(meta["chunks"]),
        "uploaded_at": str(meta.get("uploaded_at") or ""),
    }


def _metadata_from_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "filename": str(rec["filename"]),
        "stored_name": str(rec["stored_name"]),
        "path": str(rec["path"]),
        "sha256": str(rec["sha256"]),
        "collection_name": str(rec["collection_name"]),
        "chunks": int(rec["chunks"]),
        "uploaded_at": str(rec.get("uploaded_at") or ""),
    }


def _qdrant_recreate_registry_collection(client: Any) -> None:
    from qdrant_client.models import Distance, VectorParams

    name = STREAMLIT_REGISTRY_COLLECTION
    if client.collection_exists(collection_name=name):
        client.delete_collection(collection_name=name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


def load_streamlit_file_registry(client: Any, migrate_from: Optional[Path] = None) -> List[Dict[str, Any]]:
    if _is_qdrant(client):
        name = STREAMLIT_REGISTRY_COLLECTION
        if not client.collection_exists(collection_name=name):
            records: List[Dict[str, Any]] = []
        else:
            records = []
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for p in points:
                    payload = dict(p.payload or {})
                    payload.pop(DOC_PAYLOAD_KEY, None)
                    if payload:
                        records.append(_record_from_metadata(payload))
                if offset is None:
                    break
        if not records and migrate_from is not None and migrate_from.exists():
            raw = migrate_from.read_text(encoding="utf-8").strip() or "[]"
            migrated = json.loads(raw)
            if isinstance(migrated, list) and migrated:
                save_streamlit_file_registry(client, migrated)
                try:
                    migrate_from.unlink()
                except OSError:
                    pass
                return migrated
        return records

    coll = get_or_create_vector_collection(client, STREAMLIT_REGISTRY_COLLECTION)
    data = coll.get(include=["metadatas"])
    metas = data.get("metadatas") or []
    records = [_record_from_metadata(m) for m in metas if m]
    if not records and migrate_from is not None and migrate_from.exists():
        raw = migrate_from.read_text(encoding="utf-8").strip() or "[]"
        migrated = json.loads(raw)
        if isinstance(migrated, list) and migrated:
            save_streamlit_file_registry(client, migrated)
            try:
                migrate_from.unlink()
            except OSError:
                pass
            return migrated
    return records


def save_streamlit_file_registry(client: Any, records: List[Dict[str, Any]]) -> None:
    if _is_qdrant(client):
        from qdrant_client.models import PointStruct

        _qdrant_recreate_registry_collection(client)
        if not records:
            return
        name = STREAMLIT_REGISTRY_COLLECTION
        points: List[Any] = []
        for rec in records:
            rid = str(rec["sha256"])
            meta = _metadata_from_record(rec)
            payload = dict(meta)
            payload[DOC_PAYLOAD_KEY] = str(rec["filename"])
            vec = [0.0] * EMBED_DIM
            points.append(PointStruct(id=_stable_point_id(rid), vector=vec, payload=payload))
        for start in range(0, len(points), 256):
            batch = points[start : start + 256]
            client.upsert(collection_name=name, points=batch)
        return

    coll = get_or_create_vector_collection(client, STREAMLIT_REGISTRY_COLLECTION)
    existing = coll.get()
    old_ids = existing.get("ids") or []
    if old_ids:
        coll.delete(ids=old_ids)
    if not records:
        return
    ids = [str(r["sha256"]) for r in records]
    metadatas = [_metadata_from_record(r) for r in records]
    embeddings = [[0.0] * EMBED_DIM for _ in records]
    documents = [str(r["filename"]) for r in records]
    coll.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


class QdrantCollectionAdapter:
    def __init__(self, client: Any, collection_name: str) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._client = client
        self._name = collection_name
        if not self._client.collection_exists(collection_name=self._name):
            self._client.create_collection(
                collection_name=self._name,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        from qdrant_client.models import PointStruct

        points: List[Any] = []
        for i, raw_id in enumerate(ids):
            payload = dict(metadatas[i])
            payload[DOC_PAYLOAD_KEY] = documents[i]
            points.append(
                PointStruct(id=_stable_point_id(raw_id), vector=embeddings[i], payload=payload),
            )
        for start in range(0, len(points), 256):
            batch = points[start : start + 256]
            self._client.upsert(collection_name=self._name, points=batch)

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        include = include or ["documents", "distances"]
        qv = query_embeddings[0]
        resp = self._client.query_points(
            collection_name=self._name,
            query=qv,
            limit=n_results,
            with_payload=True,
        )
        hits = resp.points
        docs: List[str] = []
        distances: List[float] = []
        metas: List[Dict[str, Any]] = []
        for h in hits:
            payload = dict(h.payload or {})
            text = payload.pop(DOC_PAYLOAD_KEY, "")
            docs.append(str(text))
            # Chroma cosine distance: lower is more similar. Qdrant cosine score: higher is more similar.
            score = float(h.score)
            distances.append(max(0.0, min(2.0, 1.0 - score)))
            metas.append(payload)
        out: Dict[str, Any] = {}
        if "documents" in include:
            out["documents"] = [docs]
        if "distances" in include:
            out["distances"] = [distances]
        if "metadatas" in include:
            out["metadatas"] = [metas]
        return out

    def get(self, include: Optional[List[str]] = None) -> Dict[str, Any]:
        """Minimal Chroma-compatible surface for registry code paths (unused for Qdrant registry)."""
        _ = include
        offset = None
        ids: List[int] = []
        metadatas: List[Dict[str, Any]] = []
        while True:
            points, offset = self._client.scroll(
                collection_name=self._name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                ids.append(int(p.id))  # type: ignore[arg-type]
                payload = dict(p.payload or {})
                payload.pop(DOC_PAYLOAD_KEY, None)
                metadatas.append(payload)
            if offset is None:
                break
        return {"ids": ids, "metadatas": metadatas}


# Backwards-compatible names for imports expecting "chroma" helpers
create_chroma_client = create_vector_client
