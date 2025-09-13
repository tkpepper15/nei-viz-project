"""
Simple Lightweight Storage System
================================

A simplified version that uses JSON and SQLite instead of HDF5,
making it easier to run without additional dependencies.

This demonstrates the core concept of procedural configuration generation
and lightweight storage.
"""

import numpy as np
import json
import sqlite3
from typing import List, Dict, Tuple, Optional, Union
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


class SimpleLightweightStorage:
    """Simple lightweight storage using SQLite only"""
    
    def __init__(self, storage_dir: str = "simple_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize measurement config manager
        self.measurement_manager = MeasurementConfigManager()
    
    def _get_db_path(self, dataset_id: str) -> Path:
        """Get database path for dataset"""
        return self.storage_dir / f"{dataset_id}.db"
    
    def _create_database(self, dataset_id: str, measurement_config_name: str, grid_size: int):
        """Create SQLite database for dataset"""
        db_path = self._get_db_path(dataset_id)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                linear_index INTEGER NOT NULL,
                config_string TEXT NOT NULL,
                resnorm REAL NOT NULL,
                computation_time REAL,
                rank INTEGER,
                UNIQUE(linear_index)
            )
        ''')
        
        # Create metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        
        # Store metadata
        metadata = {
            'dataset_id': dataset_id,
            'measurement_config': measurement_config_name,
            'grid_size': str(grid_size),
            'creation_time': datetime.now().isoformat(),
            'total_configurations': str(grid_size ** 5)
        }
        
        for key, value in metadata.items():
            cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', (key, value))
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_resnorm ON results(resnorm)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rank ON results(rank)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_linear ON results(linear_index)')
        
        conn.commit()
        conn.close()
    
    def store_results(self, 
                     config_ids: List[ConfigId], 
                     resnorms: List[float],
                     measurement_config_name: str,
                     computation_times: Optional[List[float]] = None,
                     dataset_name: Optional[str] = None) -> str:
        """Store computation results"""
        
        if len(config_ids) != len(resnorms):
            raise ValueError("config_ids and resnorms must have same length")
        
        # Generate dataset ID
        grid_size = config_ids[0].grid_size if config_ids else 0
        content = f"{measurement_config_name}_{grid_size}_{datetime.now().isoformat()}"
        dataset_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Create database
        self._create_database(dataset_id, measurement_config_name, grid_size)
        
        # Prepare data with ranks
        resnorm_array = np.array(resnorms)
        sorted_indices = np.argsort(resnorm_array)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(resnorms)) + 1
        
        # Store results
        db_path = self._get_db_path(dataset_id)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Insert results
        for i, config_id in enumerate(config_ids):
            comp_time = computation_times[i] if computation_times is not None else None
            
            cursor.execute('''
                INSERT INTO results (linear_index, config_string, resnorm, computation_time, rank)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                config_id.to_linear_index(),
                config_id.to_string(),
                float(resnorms[i]),
                comp_time,
                int(ranks[i])
            ))
        
        # Update metadata with final stats
        cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', 
                      ('stored_results', str(len(config_ids))))
        cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', 
                      ('min_resnorm', str(float(resnorm_array.min()))))
        cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', 
                      ('max_resnorm', str(float(resnorm_array.max()))))
        
        # Calculate storage size
        conn.commit()
        storage_size = Path(db_path).stat().st_size / 1024**2
        cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', 
                      ('storage_size_mb', str(storage_size)))
        
        conn.commit()
        conn.close()
        
        print(f"Stored {len(config_ids):,} results as dataset '{dataset_id}'")
        print(f"Storage size: {storage_size:.2f} MB")
        print(f"Storage efficiency: {len(config_ids)}/{grid_size**5} = {len(config_ids)/(grid_size**5)*100:.1f}% of full grid")
        
        return dataset_id
    
    def load_results(self, dataset_id: str, 
                    max_results: Optional[int] = None,
                    min_resnorm: Optional[float] = None,
                    max_resnorm: Optional[float] = None) -> List[StoredResult]:
        """Load stored results with optional filtering"""
        
        db_path = self._get_db_path(dataset_id)
        if not db_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_id} not found")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get grid size from metadata
        cursor.execute('SELECT value FROM metadata WHERE key = ?', ('grid_size',))
        grid_size = int(cursor.fetchone()[0])
        
        # Build query
        query = "SELECT linear_index, config_string, resnorm, computation_time, rank FROM results WHERE 1=1"
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
        rows = cursor.fetchall()
        
        # Get creation time
        cursor.execute('SELECT value FROM metadata WHERE key = ?', ('creation_time',))
        creation_time = cursor.fetchone()[0]
        
        conn.close()
        
        # Convert to StoredResult objects
        results = []
        for linear_index, config_string, resnorm, comp_time, rank in rows:
            # Ensure linear_index is properly converted to int
            linear_index = int(linear_index) if linear_index is not None else 0
            config_id = ConfigId.from_linear_index(linear_index, grid_size)
            
            result = StoredResult(
                config_id=config_id,
                resnorm=resnorm,
                computation_time=comp_time,
                timestamp=creation_time
            )
            results.append(result)
        
        return results
    
    def get_best_results(self, dataset_id: str, n_best: int = 100) -> List[Tuple[ConfigId, CircuitParameters, float]]:
        """Get the best N results with full parameter expansion"""
        
        # Get best results from storage
        stored_results = self.load_results(dataset_id, max_results=n_best)
        
        # Get grid size
        db_path = self._get_db_path(dataset_id)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT value FROM metadata WHERE key = ?', ('grid_size',))
        grid_size = int(cursor.fetchone()[0])
        conn.close()
        
        # Create serializer for parameter expansion
        serializer = ConfigSerializer(grid_size=grid_size)
        
        # Expand to full parameters
        expanded_results = []
        for result in stored_results:
            params = serializer.deserialize_config(result.config_id)
            expanded_results.append((result.config_id, params, result.resnorm))
        
        return expanded_results
    
    def get_metadata(self, dataset_id: str) -> Dict:
        """Get dataset metadata"""
        db_path = self._get_db_path(dataset_id)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT key, value FROM metadata')
        metadata = dict(cursor.fetchall())
        
        conn.close()
        
        # Convert numeric values
        for key in ['grid_size', 'total_configurations', 'stored_results']:
            if key in metadata:
                metadata[key] = int(metadata[key])
        
        for key in ['min_resnorm', 'max_resnorm', 'storage_size_mb']:
            if key in metadata:
                metadata[key] = float(metadata[key])
        
        return metadata
    
    def list_datasets(self) -> List[Tuple[str, Dict]]:
        """List all stored datasets"""
        datasets = []
        
        for db_file in self.storage_dir.glob("*.db"):
            dataset_id = db_file.stem
            try:
                metadata = self.get_metadata(dataset_id)
                datasets.append((dataset_id, metadata))
            except Exception as e:
                print(f"Error reading {dataset_id}: {e}")
        
        return datasets
    
    def get_storage_stats(self) -> Dict:
        """Get overall storage statistics"""
        datasets = self.list_datasets()
        
        if not datasets:
            return {"total_datasets": 0, "total_size_mb": 0}
        
        total_size = sum(metadata.get('storage_size_mb', 0) for _, metadata in datasets)
        total_results = sum(metadata.get('stored_results', 0) for _, metadata in datasets)
        
        return {
            "total_datasets": len(datasets),
            "total_results": total_results,
            "total_size_mb": total_size,
            "average_size_mb": total_size / len(datasets) if datasets else 0,
            "datasets": [
                {
                    "id": dataset_id,
                    "measurement_config": metadata.get('measurement_config', 'unknown'),
                    "grid_size": metadata.get('grid_size', 0),
                    "results": metadata.get('stored_results', 0),
                    "size_mb": metadata.get('storage_size_mb', 0),
                    "compression_ratio": metadata.get('total_configurations', 1) / max(metadata.get('stored_results', 1), 1)
                }
                for dataset_id, metadata in datasets
            ]
        }
    
    def delete_dataset(self, dataset_id: str):
        """Delete a stored dataset"""
        db_path = self._get_db_path(dataset_id)
        if db_path.exists():
            db_path.unlink()
            print(f"Deleted dataset {dataset_id}")
        else:
            print(f"Dataset {dataset_id} not found")


def demo_simple_storage():
    """Demonstrate the simple storage system"""
    print("=== Simple Lightweight Storage Demo ===\n")
    
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
    storage = SimpleLightweightStorage("demo_simple_storage")
    
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
    
    # Demonstrate filtering
    print(f"\n=== Filtering Demonstration ===")
    
    # Get results with resnorm < 1.0
    filtered_results = storage.load_results(dataset_id, max_resnorm=1.0, max_results=20)
    print(f"Results with resnorm < 1.0: {len(filtered_results)}")
    
    # Show some filtered results
    if filtered_results:
        print("Sample filtered results:")
        for i, result in enumerate(filtered_results[:5]):
            params = serializer.deserialize_config(result.config_id)
            print(f"  {i+1}. Resnorm: {result.resnorm:.6f}, Config: {result.config_id.to_string()}")
    
    # Get metadata
    metadata = storage.get_metadata(dataset_id)
    print(f"\nDataset metadata:")
    print(f"  Grid size: {metadata['grid_size']}")
    print(f"  Total configurations: {metadata['total_configurations']:,}")
    print(f"  Stored results: {metadata['stored_results']:,}")
    print(f"  Resnorm range: {metadata['min_resnorm']:.6f} - {metadata['max_resnorm']:.6f}")
    print(f"  Storage efficiency: {metadata['stored_results']/metadata['total_configurations']*100:.1f}% of full grid")
    
    # Demonstrate config regeneration
    print(f"\n=== Configuration Regeneration Demo ===")
    
    # Take a config ID and show how we can regenerate everything
    sample_result = best_results[0]
    config_id, params, resnorm = sample_result
    
    print(f"Sample Config ID: {config_id.to_string()}")
    print(f"Linear Index: {config_id.to_linear_index()}")
    print(f"Regenerated Parameters:")
    print(f"  Rsh: {params.rsh:.2e} Ω")
    print(f"  Ra:  {params.ra:.2e} Ω")
    print(f"  Ca:  {params.ca:.2e} F ({params.ca*1e6:.1f} μF)")
    print(f"  Rb:  {params.rb:.2e} Ω")
    print(f"  Cb:  {params.cb:.2e} F ({params.cb*1e6:.1f} μF)")
    print(f"Associated Resnorm: {resnorm:.6f}")
    
    return dataset_id, storage


if __name__ == "__main__":
    demo_simple_storage()