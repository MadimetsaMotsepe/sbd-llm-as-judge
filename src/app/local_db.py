"""
Local database implementation for development purposes.
Uses JSON file storage instead of Cosmos DB.
"""

import json
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4


class LocalDatabase:
    """Simple JSON-based database for local development."""
    
    def __init__(self, db_path: str = "./local_db.json"):
        self.db_path = db_path
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Default structure
        return {
            "judges": {},
            "assemblies": {}
        }
    
    # NEW: recursive sanitizer for non-serializable objects (e.g., HttpUrl, Pydantic models)
    def _sanitize(self, obj: Any) -> Any:  # noqa: D401
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        # Pydantic models
        if hasattr(obj, "model_dump"):
            try:
                return self._sanitize(obj.model_dump())
            except Exception:  # pragma: no cover - defensive
                return str(obj)
        # Fallback: convert to string (covers HttpUrl / Url types)
        try:
            return str(obj)
        except Exception:  # pragma: no cover - defensive
            return repr(obj)
    
    def _save_data(self) -> None:
        """Save data to JSON file (with sanitization)."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        sanitized = self._sanitize(self.data)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(sanitized, f, indent=2, ensure_ascii=False)
    
    # Judge operations
    async def list_judges(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all judges, optionally filtered by name."""
        judges = list(self.data["judges"].values())
        if name:
            judges = [j for j in judges if name.lower() in j.get("name", "").lower()]
        return judges
    
    async def create_judge(self, judge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new judge."""
        if "id" not in judge_data:
            judge_data["id"] = str(uuid4())
        
        self.data["judges"][judge_data["id"]] = judge_data
        self._save_data()
        return judge_data
    
    async def get_judge(self, judge_id: str) -> Optional[Dict[str, Any]]:
        """Get a judge by ID."""
        return self.data["judges"].get(judge_id)
    
    async def update_judge(self, judge_id: str, judge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing judge."""
        if judge_id not in self.data["judges"]:
            raise KeyError(f"Judge {judge_id} not found")
        
        existing = self.data["judges"][judge_id]
        updated = {**existing, **judge_data}
        self.data["judges"][judge_id] = updated
        self._save_data()
        return updated
    
    async def delete_judge(self, judge_id: str) -> bool:
        """Delete a judge."""
        if judge_id in self.data["judges"]:
            del self.data["judges"][judge_id]
            self._save_data()
            return True
        return False
    
    # Assembly operations
    async def list_assemblies(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all assemblies, optionally filtered by role."""
        assemblies = list(self.data["assemblies"].values())
        if role:
            assemblies = [a for a in assemblies if role in a.get("roles", [])]
        return assemblies
    
    async def create_assembly(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new assembly."""
        if "id" not in assembly_data:
            assembly_data["id"] = str(uuid4())
        
        self.data["assemblies"][assembly_data["id"]] = assembly_data
        self._save_data()
        return assembly_data
    
    async def get_assembly(self, assembly_id: str) -> Optional[Dict[str, Any]]:
        """Get an assembly by ID."""
        return self.data["assemblies"].get(assembly_id)
    
    async def update_assembly(self, assembly_id: str, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing assembly."""
        if assembly_id not in self.data["assemblies"]:
            raise KeyError(f"Assembly {assembly_id} not found")
        
        existing = self.data["assemblies"][assembly_id]
        updated = {**existing, **assembly_data}
        self.data["assemblies"][assembly_id] = updated
        self._save_data()
        return updated
    
    async def delete_assembly(self, assembly_id: str) -> bool:
        """Delete an assembly."""
        if assembly_id in self.data["assemblies"]:
            del self.data["assemblies"][assembly_id]
            self._save_data()
            return True
        return False


# Global instance
_local_db = None

def get_local_db() -> LocalDatabase:
    """Get the global local database instance."""
    global _local_db
    if _local_db is None:
        db_path = os.getenv("LOCAL_DB_PATH", "./local_db.json")
        _local_db = LocalDatabase(db_path)
    return _local_db
