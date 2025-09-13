"""
Lightweight Storage System for Circuit Configurations
====================================================

This system stores only essential data:
- Configuration IDs (compact parameter indices)
- Computed resnorms
- Measurement config references

Full configurations are regenerated procedurally when needed,
dramatically reducing storage requirements.

Usage:
    storage = LightweightStorage()
    storage.store_results(config_ids, resnorms, measurement_config_name)
    results = storage.load_results(measurement_config_name)
"""

import numpy as np
import json
import sqlite3
import h5py
from typing import List, Dict, Tuple, Optional, Union, Iterator
from pathlib import Path
from dataclasses import dataclass, asdict
import time
import hashlib
from datetime import datetime

from config_serializer import ConfigId, ConfigSerializer, CircuitParameters
from measurement_config import MeasurementConfig, MeasurementConfigManager


@dataclass
class StoredResult:
    """A single stored computation result"""
    config_id: ConfigId
    resnorm: float
    computation_time: Optional[float] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'config_string': self.config_id.to_string(),
            'linear_index': self.config_id.to_linear_index(),
            'resnorm': self.resnorm,
            'computation_time': self.computation_time,
            'timestamp': self.timestamp
        }


@dataclass
class StorageMetadata:
    """Metadata for a stored dataset"""
    measurement_config_name: str
    grid_size: int
    total_configurations: int
    stored_results: int
    min_resnorm: float
    max_resnorm: float
    computation_date: str
    storage_size_mb: float
    file_hash: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LightweightStorage:
    """Manages lightweight storage of configuration results"""
    
    def __init__(self, storage_dir: str = "lightweight_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize measurement config manager
        self.measurement_manager = MeasurementConfigManager()
        
        # Create storage subdirectories
        (self.storage_dir / "results").mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
        (self.storage_dir / "indices").mkdir(exist_ok=True)
    
    def _generate_dataset_id(self, measurement_config_name: str, grid_size: int) -> str:
        """Generate unique dataset identifier"""
        content = f"{measurement_config_name}_{grid_size}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_filepath(self, dataset_id: str, file_type: str) -> Path:
        """Get filepath for different storage components"""
        if file_type == "results":
            return self.storage_dir / "results" / f"{dataset_id}.h5"
        elif file_type == "metadata":
            return self.storage_dir / "metadata" / f"{dataset_id}.json"
        elif file_type == "index":
            return self.storage_dir / "indices" / f"{dataset_id}.db"
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def store_results(self, 
                     config_ids: List[ConfigId], 
                     resnorms: List[float],
                     measurement_config_name: str,
                     computation_times: Optional[List[float]] = None,
                     dataset_name: Optional[str] = None) -> str:
        """
        Store computation results in lightweight format
        
        Returns:
            dataset_id: Unique identifier for the stored dataset
        """
        
        if len(config_ids) != len(resnorms):
            raise ValueError("config_ids and resnorms must have same length")
        
        # Generate dataset ID
        grid_size = config_ids[0].grid_size if config_ids else 0
        dataset_id = self._generate_dataset_id(measurement_config_name, grid_size)
        
        if dataset_name is None:
            dataset_name = f"Dataset_{dataset_id}"
        
        # Prepare data
        n_results = len(config_ids)
        linear_indices = np.array([cid.to_linear_index() for cid in config_ids])
        resnorm_array = np.array(resnorms)
        
        if computation_times is not None:
            comp_time_array = np.array(computation_times)
        else:
            comp_time_array = np.full(n_results, np.nan)
        
        # Store in HDF5 for efficient binary storage
        results_path = self._get_filepath(dataset_id, "results")
        
        with h5py.File(results_path, 'w') as f:
            # Store core data
            f.create_dataset('linear_indices', data=linear_indices, 
                           compression='gzip', compression_opts=9)
            f.create_dataset('resnorms', data=resnorm_array,
                           compression='gzip', compression_opts=9)
            f.create_dataset('computation_times', data=comp_time_array,
                           compression='gzip', compression_opts=9)
            
            # Store metadata as attributes
            f.attrs['dataset_id'] = dataset_id
            f.attrs['dataset_name'] = dataset_name
            f.attrs['measurement_config'] = measurement_config_name
            f.attrs['grid_size'] = grid_size
            f.attrs['n_results'] = n_results
            f.attrs['creation_time'] = datetime.now().isoformat()
        
        # Create SQLite index for fast queries
        self._create_sqlite_index(dataset_id, linear_indices, resnorm_array)
        
        # Store metadata
        self._store_metadata(dataset_id, measurement_config_name, grid_size, 
                           n_results, resnorm_array, results_path)
        
        print(f"Stored {n_results:,} results as dataset '{dataset_id}'")
        print(f"Storage size: {results_path.stat().st_size / 1024**2:.2f} MB")
        
        return dataset_id
    
    def _create_sqlite_index(self, dataset_id: str, linear_indices: np.ndarray, 
                           resnorms: np.ndarray):
        """Create SQLite index for fast querying"""
        db_path = self._get_filepath(dataset_id, "index")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE results (
                linear_index INTEGER PRIMARY KEY,
                resnorm REAL NOT NULL,
                rank INTEGER
            )
        ''')
        
        # Sort by resnorm and assign ranks
        sorted_indices = np.argsort(resnorms)
        
        # Insert data with ranks
        data = [(int(linear_indices[i]), float(resnorms[i]), int(rank+1)) 
                for rank, i in enumerate(sorted_indices)]
        
        cursor.executemany('INSERT INTO results VALUES (?, ?, ?)', data)
        
        # Create indices for fast queries
        cursor.execute('CREATE INDEX idx_resnorm ON results(resnorm)')
        cursor.execute('CREATE INDEX idx_rank ON results(rank)')
        
        conn.commit()
        conn.close()
    
    def _store_metadata(self, dataset_id: str, measurement_config_name: str,
                       grid_size: int, n_results: int, resnorms: np.ndarray,
                       results_path: Path):
        """Store dataset metadata"""
        
        # Calculate file hash
        with open(results_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        metadata = StorageMetadata(
            measurement_config_name=measurement_config_name,
            grid_size=grid_size,
            total_configurations=grid_size**5,
            stored_results=n_results,
            min_resnorm=float(resnorms.min()),
            max_resnorm=float(resnorms.max()),
            computation_date=datetime.now().isoformat(),
            storage_size_mb=results_path.stat().st_size / 1024**2,
            file_hash=file_hash
        )
        
        metadata_path = self._get_filepath(dataset_id, "metadata")
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def load_results(self, dataset_id: str, 
                    max_results: Optional[int] = None,
                    min_resnorm: Optional[float] = None,
                    max_resnorm: Optional[float] = None) -> List[StoredResult]:
        """Load stored results with optional filtering"""
        
        # Check if dataset exists
        results_path = self._get_filepath(dataset_id, "results")
        if not results_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_id} not found")
        
        # Load from HDF5
        with h5py.File(results_path, 'r') as f:
            linear_indices = f['linear_indices'][:]
            resnorms = f['resnorms'][:]
            computation_times = f['computation_times'][:]
            
            grid_size = f.attrs['grid_size']
            creation_time = f.attrs['creation_time']
        
        # Apply filtering via SQLite for efficiency
        if min_resnorm is not None or max_resnorm is not None or max_results is not None:
            filtered_indices = self._query_sqlite_index(dataset_id, max_results, 
                                                       min_resnorm, max_resnorm)
            
            # Filter the full arrays
            mask = np.isin(linear_indices, filtered_indices)
            linear_indices = linear_indices[mask]
            resnorms = resnorms[mask]
            computation_times = computation_times[mask]
        
        # Convert back to StoredResult objects
        results = []
        for i in range(len(linear_indices)):
            config_id = ConfigId.from_linear_index(int(linear_indices[i]), grid_size)
            
            comp_time = float(computation_times[i]) if not np.isnan(computation_times[i]) else None
            
            result = StoredResult(
                config_id=config_id,
                resnorm=float(resnorms[i]),
                computation_time=comp_time,
                timestamp=creation_time
            )
            results.append(result)
        
        return results
    
    def _query_sqlite_index(self, dataset_id: str, max_results: Optional[int],
                          min_resnorm: Optional[float], max_resnorm: Optional[float]) -> List[int]:
        """Query SQLite index for filtered linear indices"""
        
        db_path = self._get_filepath(dataset_id, "index")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT linear_index FROM results WHERE 1=1"
        params = []
        
        if min_resnorm is not None:
            query += " AND resnorm >= ?"
            params.append(min_resnorm)
        
        if max_resnorm is not None:
            query += " AND resnorm <= ?"
            params.append(max_resnorm)
        
        query += " ORDER BY resnorm"
        
        if max_results is not None:
            query += " LIMIT ?"
            params.append(max_results)
        
        cursor.execute(query, params)
        indices = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return indices
    
    def get_best_results(self, dataset_id: str, n_best: int = 100) -> List[Tuple[ConfigId, CircuitParameters, float]]:
        """Get the best N results with full parameter expansion"""
        
        # Get best results from storage
        stored_results = self.load_results(dataset_id, max_results=n_best)
        
        # Load metadata to get grid size
        metadata = self.get_metadata(dataset_id)
        
        # Create serializer for parameter expansion
        serializer = ConfigSerializer(grid_size=metadata.grid_size)
        
        # Expand to full parameters
        expanded_results = []
        for result in stored_results:
            params = serializer.deserialize_config(result.config_id)
            expanded_results.append((result.config_id, params, result.resnorm))
        
        return expanded_results
    
    def get_metadata(self, dataset_id: str) -> StorageMetadata:
        """Get dataset metadata"""
        metadata_path = self._get_filepath(dataset_id, "metadata")
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        return StorageMetadata(**data)
    
    def list_datasets(self) -> List[Tuple[str, StorageMetadata]]:
        """List all stored datasets"""
        datasets = []
        
        for metadata_file in (self.storage_dir / "metadata").glob("*.json"):
            dataset_id = metadata_file.stem
            metadata = self.get_metadata(dataset_id)
            datasets.append((dataset_id, metadata))
        
        return datasets
    
    def get_storage_stats(self) -> Dict:
        """Get overall storage statistics"""
        datasets = self.list_datasets()
        
        if not datasets:
            return {"total_datasets": 0, "total_size_mb": 0}
        
        total_size = sum(metadata.storage_size_mb for _, metadata in datasets)
        total_results = sum(metadata.stored_results for _, metadata in datasets)
        
        return {
            "total_datasets": len(datasets),
            "total_results": total_results,
            "total_size_mb": total_size,
            "average_size_mb": total_size / len(datasets),
            "datasets": [
                {
                    "id": dataset_id,
                    "name": metadata.measurement_config_name,
                    "grid_size": metadata.grid_size,
                    "results": metadata.stored_results,
                    "size_mb": metadata.storage_size_mb,
                    "compression_ratio": metadata.total_configurations / metadata.stored_results
                }
                for dataset_id, metadata in datasets
            ]
        }
    
    def delete_dataset(self, dataset_id: str):
        """Delete a stored dataset"""
        for file_type in ["results", "metadata", "index"]:
            filepath = self._get_filepath(dataset_id, file_type)
            if filepath.exists():
                filepath.unlink()
        
        print(f"Deleted dataset {dataset_id}")


def demo_lightweight_storage():
    """Demonstrate the lightweight storage system"""
    print("=== Lightweight Storage Demo ===\n")
    
    # Create sample data
    print("Generating sample configuration data...")
    serializer = ConfigSerializer(grid_size=5)  # Small grid for demo
    
    # Generate all possible configurations
    all_configs = serializer.generate_all_configs()
    
    # Simulate computation results (random resnorms for demo)
    np.random.seed(42)  # Reproducible results
    config_ids = [config_id for config_id, _ in all_configs]
    resnorms = np.random.exponential(1.0, len(config_ids))  # Exponential distribution
    computation_times = np.random.uniform(0.1, 2.0, len(config_ids))  # Random compute times
    
    print(f"Generated {len(config_ids)} configurations with simulated resnorms")
    
    # Create storage system
    storage = LightweightStorage("demo_storage")
    
    # Store results
    dataset_id = storage.store_results(
        config_ids=config_ids,
        resnorms=resnorms,
        measurement_config_name="standard_eis",
        computation_times=computation_times,
        dataset_name="Demo Grid 5 Results"
    )
    
    print(f"\nStored as dataset: {dataset_id}")
    
    # Load and analyze results
    print(f"\n=== Data Analysis ===")
    
    # Get best results
    best_results = storage.get_best_results(dataset_id, n_best=10)
    print(f"\nTop 10 Best Results:")
    for i, (config_id, params, resnorm) in enumerate(best_results):
        print(f"  {i+1:2d}. Resnorm: {resnorm:.6f}")
        print(f"      Config: {config_id.to_string()}")
        print(f"      Rsh: {params.rsh:.0f}Ω, Ra: {params.ra:.0f}Ω, Ca: {params.ca*1e6:.1f}μF")
    
    # Storage statistics
    stats = storage.get_storage_stats()
    print(f"\n=== Storage Statistics ===")
    print(f"Total datasets: {stats['total_datasets']}")
    print(f"Total results: {stats['total_results']:,}")
    print(f"Total storage: {stats['total_size_mb']:.2f} MB")
    print(f"Avg storage per dataset: {stats['average_size_mb']:.2f} MB")
    
    # Demonstrate filtering
    print(f"\n=== Filtering Demonstration ===")
    
    # Get results with resnorm < 1.0
    filtered_results = storage.load_results(dataset_id, max_resnorm=1.0, max_results=20)
    print(f"Results with resnorm < 1.0: {len(filtered_results)}")
    
    # Get metadata
    metadata = storage.get_metadata(dataset_id)
    print(f"\nDataset metadata:")
    print(f"  Grid size: {metadata.grid_size}")
    print(f"  Total configurations: {metadata.total_configurations:,}")
    print(f"  Stored results: {metadata.stored_results:,}")
    print(f"  Resnorm range: {metadata.min_resnorm:.6f} - {metadata.max_resnorm:.6f}")
    print(f"  Storage efficiency: {metadata.stored_results/metadata.total_configurations*100:.1f}% of full grid")
    
    return dataset_id, storage


if __name__ == "__main__":
    demo_lightweight_storage()