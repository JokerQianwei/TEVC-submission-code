from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
from tdc import Oracle

_QED, _SA = Oracle(name='QED'), Oracle(name='SA')

def compute_qed_sa(smiles: str) -> Tuple[Optional[float], Optional[float]]:
    try: return float(_QED(smiles)), float(_SA(smiles))
    except: return None, None

@dataclass
class ChemMetricCache:
    cache_path: Optional[Path]
    _data: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict, init=False)
    _dirty: bool = field(default=False, init=False)

    def __post_init__(self):
        if not self.cache_path: return
        self.cache_path = Path(self.cache_path)
        if self.cache_path.exists():
            try: self._data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except: self._data = {}

    def get(self, smiles: str) -> Tuple[Optional[float], Optional[float]]:
        return (d.get("qed"), d.get("sa")) if (d := self._data.get(smiles)) else (None, None)

    def set(self, smiles: str, qed: Optional[float], sa: Optional[float]):
        self._data[smiles] = {"qed": qed, "sa": sa}
        self._dirty = True

    def get_or_compute(self, smiles: str) -> Tuple[Optional[float], Optional[float]]:
        if smiles in self._data: return self.get(smiles)
        qed, sa = compute_qed_sa(smiles)
        self.set(smiles, qed, sa)
        return qed, sa

    def flush(self):
        if not self.cache_path or not self._dirty: return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp_{os.getpid()}")
        try:
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.cache_path)
            self._dirty = False
        except: 
            if tmp.exists(): tmp.unlink()
