from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Iterator
import pandas as pd
import numpy as np
import time
import logging
from collections import defaultdict, Counter
import threading
import queue
import hashlib
from dataclasses import dataclass
import psutil
import gc
from itertools import islice
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NodeBatchMetrics:
    """Track performance metrics for node insertion adaptive optimization"""
    node_type: str
    batch_size: int
    records: int
    duration: float
    success_rate: float
    properties_count: int
    
class NodeBatchOptimizer:
    """Dynamically optimize node batch sizes based on performance feedback"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.optimal_sizes = {}
        
    def record_batch_performance(self, metrics: NodeBatchMetrics):
        """Record batch performance for learning"""
        self.metrics_history[metrics.node_type].append(metrics)
        
    def get_optimal_batch_size(self, node_type: str, properties_count: int, default_size: int = 2000) -> int:
        """Calculate optimal batch size based on historical performance and complexity"""
        
        # Adjust default based on property complexity
        complexity_adjusted_default = max(500, default_size - (properties_count * 200))
        
        if node_type not in self.metrics_history:
            return complexity_adjusted_default
            
        history = self.metrics_history[node_type][-15:]  # Last 15 batches
        if len(history) < 3:
            return complexity_adjusted_default
            
        # Calculate throughput (records/second) for different batch sizes
        throughput_by_size = defaultdict(list)
        for metric in history:
            throughput = metric.records / metric.duration if metric.duration > 0 else 0
            # Weight by success rate
            weighted_throughput = throughput * (metric.success_rate / 100.0)
            throughput_by_size[metric.batch_size].append(weighted_throughput)
        
        # Find batch size with best average weighted throughput
        best_size = complexity_adjusted_default
        best_throughput = 0
        
        for size, throughputs in throughput_by_size.items():
            avg_throughput = sum(throughputs) / len(throughputs)
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_size = size
                
        return best_size

class IntelligentNodeChunker:
    """Advanced node chunking strategies"""
    
    def __init__(self, optimizer: NodeBatchOptimizer):
        self.optimizer = optimizer
        
    def analyze_node_data_pattern(self, df: pd.DataFrame, key_col: str) -> Dict:
        """Analyze node data patterns for optimal chunking"""
        
        total_records = len(df)
        unique_keys = df[key_col].nunique()
        
        # Check for duplicate keys (shouldn't exist but let's be safe)
        duplicate_rate = (total_records - unique_keys) / total_records if total_records > 0 else 0
        
        # Analyze data distribution
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        avg_properties_per_record = len(df.columns) - 1  # Exclude key column
        
        # Check for null patterns
        null_counts = df.isnull().sum()
        high_null_columns = null_counts[null_counts > total_records * 0.3].index.tolist()
        
        analysis = {
            'total_records': total_records,
            'unique_keys': unique_keys,
            'duplicate_rate': duplicate_rate,
            'memory_usage_mb': memory_usage,
            'avg_properties': avg_properties_per_record,
            'high_null_columns': high_null_columns,
            'complexity_score': self._calculate_complexity_score(df)
        }
        
        return analysis
    
    def _calculate_complexity_score(self, df: pd.DataFrame) -> float:
        """Calculate data complexity score for chunking strategy"""
        
        complexity = 0.0
        
        # Base complexity from number of columns
        complexity += len(df.columns) * 0.1
        
        # Add complexity for text columns (likely to have variable length)
        for col in df.columns:
            if df[col].dtype == 'object':
                complexity += 0.5
                # Add more for long text fields
                if df[col].astype(str).str.len().mean() > 100:
                    complexity += 1.0
        
        # Add complexity for null values (require handling)
        null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        complexity += null_ratio * 2
        
        return complexity
    
    def create_node_chunks(self, df: pd.DataFrame, key_col: str, node_type: str, 
                          base_batch_size: int = 2000) -> List[pd.DataFrame]:
        """Create optimized chunks for node insertion"""
        
        analysis = self.analyze_node_data_pattern(df, key_col)
        optimal_batch_size = self.optimizer.get_optimal_batch_size(
            node_type, analysis['avg_properties'], base_batch_size
        )
        
        # Adjust batch size based on complexity and memory usage
        if analysis['memory_usage_mb'] > 100:  # Large dataset
            optimal_batch_size = min(optimal_batch_size, 1000)
        elif analysis['complexity_score'] > 5:  # Complex data
            optimal_batch_size = min(optimal_batch_size, 1500)
        
        logger.info(f"Chunking {node_type}: {len(df)} records, complexity={analysis['complexity_score']:.1f}, "
                   f"batch_size={optimal_batch_size}")
        
        # Choose chunking strategy based on data characteristics
        if len(df) < 5000:
            return self._simple_node_chunk(df, optimal_batch_size)
        elif analysis['complexity_score'] > 3:
            return self._memory_efficient_chunk(df, optimal_batch_size)
        else:
            return self._performance_optimized_chunk(df, optimal_batch_size)
    
    def _simple_node_chunk(self, df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
        """Simple chunking for small datasets"""
        return [df.iloc[i:i+batch_size].copy() for i in range(0, len(df), batch_size)]
    
    def _memory_efficient_chunk(self, df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
        """Memory-efficient chunking for complex data"""
        # Smaller batches to manage memory better
        adjusted_batch_size = max(500, batch_size // 2)
        
        chunks = []
        for i in range(0, len(df), adjusted_batch_size):
            chunk = df.iloc[i:i+adjusted_batch_size].copy()
            # Optimize data types in chunk
            chunk = self._optimize_chunk_dtypes(chunk)
            chunks.append(chunk)
            
        return chunks
    
    def _performance_optimized_chunk(self, df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
        """Performance-optimized chunking for large, simple datasets"""
        # Larger batches for better throughput
        adjusted_batch_size = min(batch_size * 2, 5000)
        
        chunks = []
        for i in range(0, len(df), adjusted_batch_size):
            chunk = df.iloc[i:i+adjusted_batch_size].copy()
            chunks.append(chunk)
            
        return chunks
    
    def _optimize_chunk_dtypes(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types within a chunk to reduce memory usage"""
        
        for col in chunk.columns:
            if chunk[col].dtype == 'object':
                # Try to convert to more efficient types
                try:
                    # Try numeric conversion first
                    chunk[col] = pd.to_numeric(chunk[col], errors='ignore')
                    if chunk[col].dtype == 'object':
                        # Convert to category if reasonable number of unique values
                        if chunk[col].nunique() / len(chunk) < 0.5:
                            chunk[col] = chunk[col].astype('category')
                except:
                    pass  # Keep as object if conversion fails
                    
        return chunk

class OptimizedNeo4jNodeWriter:
    """High-performance Neo4j node writer"""
    
    def __init__(self, driver, max_connections: int = 6):
        self.driver = driver
        self.max_connections = max_connections
        self.connection_pool = queue.Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        
        # Pre-warm connections
        self._warm_connections()
    
    def _warm_connections(self):
        """Pre-establish database connections"""
        for _ in range(self.max_connections):
            try:
                session = self.driver.session()
                session.run("RETURN 1 as test").single()
                self.connection_pool.put(session)
            except Exception as e:
                logger.warning(f"Failed to pre-warm connection: {e}")
    
    def get_session(self):
        """Get a session from the pool"""
        try:
            return self.connection_pool.get(timeout=1)
        except queue.Empty:
            return self.driver.session()
    
    def return_session(self, session):
        """Return session to pool"""
        try:
            self.connection_pool.put(session, timeout=0.1)
        except queue.Full:
            session.close()
    
    def execute_node_batch(self, cypher_query: str, batch_data: List[Dict], 
                          node_type: str) -> Dict:
        """Execute node batch with optimized session management"""
        
        session = None
        start_time = time.time()
        
        try:
            session = self.get_session()
            
            total_processed = 0
            
            # For very large batches, use sub-batching
            if len(batch_data) > 3000:
                sub_batch_size = 1500
                for i in range(0, len(batch_data), sub_batch_size):
                    sub_batch = batch_data[i:i+sub_batch_size]
                    result = session.run(cypher_query, rows=sub_batch)
                    
                    # Process result
                    try:
                        summary = result.single()
                        if summary and 'processed' in summary:
                            total_processed += summary['processed']
                        else:
                            total_processed += len(sub_batch)
                    except:
                        total_processed += len(sub_batch)
                        
            else:
                # Single batch execution
                result = session.run(cypher_query, rows=batch_data)
                try:
                    summary = result.single()
                    if summary and 'processed' in summary:
                        total_processed = summary['processed']
                    else:
                        total_processed = len(batch_data)
                except:
                    total_processed = len(batch_data)
            
            duration = time.time() - start_time
            
            return {
                'success': True,
                'processed': total_processed,
                'failed': len(batch_data) - total_processed,
                'duration': duration,
                'records_per_second': total_processed / duration if duration > 0 else 0
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Node batch execution failed for {node_type}: {str(e)}")
            
            return {
                'success': False,
                'processed': 0,
                'failed': len(batch_data),
                'duration': duration,
                'error': str(e)
            }
            
        finally:
            if session:
                self.return_session(session)

class UltraOptimizedNodeInjectionSystem:
    """Ultra-optimized node injection system"""
    
    def __init__(self, driver, output_dir: str, initial_batch_size: int = 2000, max_workers: int = None):
        self.driver = driver
        self.output_dir = output_dir
        self.initial_batch_size = initial_batch_size
        
        # Auto-detect optimal worker count
        if max_workers is None:
            cpu_count = psutil.cpu_count()
            self.max_workers = min(cpu_count, 6)  # Nodes can handle more concurrency than relationships
        else:
            self.max_workers = max_workers
            
        self.optimizer = NodeBatchOptimizer()
        self.chunker = IntelligentNodeChunker(self.optimizer)
        self.writer = OptimizedNeo4jNodeWriter(driver, max_connections=self.max_workers)
        
        self.stats = defaultdict(int)
        self.lock = threading.Lock()
        
        logger.info(f"Node injection system initialized with {self.max_workers} workers")
    
    def generate_optimized_node_cypher(self, node_schema: Dict) -> str:
        """Generate highly optimized Cypher query for node creation"""
        
        node_name = node_schema['name']
        key_field = node_schema['key']
        properties = node_schema.get('properties', [])
        
        # Build property assignments, handling potential null values
        prop_assignments = []
        for prop in properties:
            prop_assignments.append(f"{prop}: row.{prop}")
        
        # Create the property string
        if prop_assignments:
            property_str = ", " + ", ".join(prop_assignments)
        else:
            property_str = ""
        
        # Optimized query using MERGE for upsert behavior
        cypher = f"""
        UNWIND $rows AS row
        MERGE (n:{node_name} {{{key_field}: row.{key_field}}})
        SET n += {{
            {key_field}: row.{key_field}{property_str}
        }}
        RETURN count(n) as processed
        """
        
        return cypher.strip()
    
    def load_and_merge_node_data(self, node_schema: Dict) -> pd.DataFrame:
        """Load and merge data from multiple tables if needed"""
        
        table_names = node_schema['table_name']
        if isinstance(table_names, str):
            table_names = [table_names]
        
        all_dataframes = []
        
        for table_name in table_names:
            csv_path = f"{self.output_dir}/{table_name}.csv"
            
            try:
                # Load with optimized dtypes
                df = pd.read_csv(csv_path, engine='c', low_memory=False)
                
                if not df.empty:
                    all_dataframes.append(df)
                    logger.info(f"Loaded {len(df)} records from {table_name}")
                    
            except Exception as e:
                logger.warning(f"Failed to load {csv_path}: {str(e)}")
                continue
        
        if not all_dataframes:
            logger.warning(f"No data found for node {node_schema['name']}")
            return pd.DataFrame()
        
        # Merge dataframes if multiple tables
        if len(all_dataframes) == 1:
            merged_df = all_dataframes[0]
        else:
            # Merge on the key field
            key_field = node_schema['key']
            merged_df = all_dataframes[0]
            
            for df in all_dataframes[1:]:
                merged_df = merged_df.merge(df, on=key_field, how='outer', suffixes=('', '_dup'))
                
                # Remove duplicate columns that might arise from merging
                duplicate_cols = [col for col in merged_df.columns if col.endswith('_dup')]
                merged_df = merged_df.drop(columns=duplicate_cols)
        
        # Remove duplicates based on key field
        key_field = node_schema['key']
        initial_size = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=[key_field])
        
        if len(merged_df) < initial_size:
            logger.info(f"Removed {initial_size - len(merged_df)} duplicate records")
        
        # Select only the columns we need
        needed_columns = [node_schema['key']] + node_schema.get('properties', [])
        available_columns = [col for col in needed_columns if col in merged_df.columns]
        merged_df = merged_df[available_columns]
        
        # Handle missing columns by adding them with null values
        missing_columns = set(needed_columns) - set(available_columns)
        for col in missing_columns:
            merged_df[col] = None
            logger.info(f"Added missing column '{col}' with null values")
        
        # Sort by key for better insertion performance
        merged_df = merged_df.sort_values(node_schema['key'])
        
        return merged_df
    
    def process_node_optimized(self, node_schema: Dict) -> Dict:
        """Process a single node type with all optimizations"""
        
        node_name = node_schema['name']
        key_field = node_schema['key']
        properties = node_schema.get('properties', [])
        
        logger.info(f"Processing {node_name} nodes with ultra-optimized strategy")
        start_time = time.time()
        
        try:
            # Load and merge data from all relevant tables
            df = self.load_and_merge_node_data(node_schema)
            
            if df.empty:
                logger.warning(f"No data found for {node_name}")
                return {'node_type': node_name, 'success': True, 'processed': 0, 'failed': 0}
            
            # Generate optimized Cypher
            cypher_query = self.generate_optimized_node_cypher(node_schema)
            
            # Create intelligent chunks
            chunks = self.chunker.create_node_chunks(df, key_field, node_name, self.initial_batch_size)
            
            logger.info(f"Created {len(chunks)} optimized chunks for {node_name}")
            
            # Process chunks with controlled concurrency
            total_processed = 0
            total_failed = 0
            
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
                
                # Submit all chunk processing tasks
                future_to_chunk = {}
                for i, chunk_df in enumerate(chunks):
                    # Convert to records, handling NaN values
                    batch_data = chunk_df.where(pd.notna(chunk_df), None).to_dict('records')
                    
                    future = executor.submit(
                        self.writer.execute_node_batch,
                        cypher_query, batch_data, f"{node_name}_chunk_{i}"
                    )
                    future_to_chunk[future] = (i, len(batch_data), len(properties))
                
                # Process completed chunks
                for future in as_completed(future_to_chunk):
                    chunk_id, chunk_size, props_count = future_to_chunk[future]
                    try:
                        result = future.result()
                        
                        if result['success']:
                            total_processed += result['processed']
                            total_failed += result['failed']
                            
                            # Record metrics for adaptive optimization
                            metrics = NodeBatchMetrics(
                                node_type=node_name,
                                batch_size=chunk_size,
                                records=result['processed'],
                                duration=result['duration'],
                                success_rate=100.0 if result['failed'] == 0 else (result['processed'] / chunk_size) * 100,
                                properties_count=props_count
                            )
                            self.optimizer.record_batch_performance(metrics)
                            
                            logger.info(f"✅ {node_name} chunk {chunk_id}: {result['processed']} processed "
                                       f"in {result['duration']:.2f}s ({result['records_per_second']:.0f} nodes/s)")
                        else:
                            total_failed += chunk_size
                            logger.error(f"❌ {node_name} chunk {chunk_id} failed: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        total_failed += chunk_size
                        logger.error(f"❌ {node_name} chunk {chunk_id} exception: {str(e)}")
            
            # Calculate final statistics
            duration = time.time() - start_time
            success_rate = (total_processed / (total_processed + total_failed)) * 100 if (total_processed + total_failed) > 0 else 0
            throughput = total_processed / duration if duration > 0 else 0
            
            logger.info(f"🎯 {node_name} completed: {total_processed} processed, {total_failed} failed, "
                       f"{success_rate:.1f}% success, {throughput:.0f} nodes/s")
            
            # Cleanup
            del df
            gc.collect()
            
            return {
                'node_type': node_name,
                'success': True,
                'processed': total_processed,
                'failed': total_failed,
                'success_rate': success_rate,
                'duration': duration,
                'throughput': throughput,
                'chunks_processed': len(chunks)
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"💥 Failed to process {node_name}: {str(e)}")
            return {
                'node_type': node_name,
                'success': False,
                'error': str(e),
                'duration': duration
            }

def execute_ultra_optimized_nodes(nodes: List, driver, output_dir: str,
                                 initial_batch_size: int = 2000, max_workers: int = None) -> Dict:
    """
    Ultra-optimized node execution with adaptive learning and intelligent concurrency
    """
    
    logger.info(f"🚀 Starting ULTRA-OPTIMIZED node injection for {len(nodes)} node types")
    start_time = time.time()
    
    # Initialize the ultra-optimized system
    system = UltraOptimizedNodeInjectionSystem(driver, output_dir, initial_batch_size, max_workers)
    
    # Process all nodes concurrently (nodes don't have dependencies like relationships)
    all_results = []
    
    with ThreadPoolExecutor(max_workers=min(len(nodes), system.max_workers)) as executor:
        
        # Submit all node processing tasks
        future_to_node = {}
        for node in nodes:
            node_dict = {
                'name': node.name,
                'key': node.key,
                'properties': node.properties,
                'table_name': node.table_name
            }
            
            future = executor.submit(system.process_node_optimized, node_dict)
            future_to_node[future] = node.name
        
        # Collect results
        for future in as_completed(future_to_node):
            node_name = future_to_node[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                logger.error(f"Node processing failed for {node_name}: {str(e)}")
                all_results.append({
                    'node_type': node_name,
                    'success': False,
                    'error': str(e)
                })
    
    # Final statistics
    total_duration = time.time() - start_time
    successful_nodes = sum(1 for r in all_results if r.get('success', False))
    total_processed = sum(r.get('processed', 0) for r in all_results)
    total_failed = sum(r.get('failed', 0) for r in all_results)
    
    overall_success_rate = (total_processed / (total_processed + total_failed)) * 100 if (total_processed + total_failed) > 0 else 0
    overall_throughput = total_processed / total_duration if total_duration > 0 else 0
    
    final_stats = {
        'total_node_types': len(nodes),
        'successful_node_types': successful_nodes,
        'total_processed': total_processed,
        'total_failed': total_failed,
        'overall_success_rate': overall_success_rate,
        'total_duration': total_duration,
        'overall_throughput': overall_throughput,
        'detailed_results': all_results
    }
    
    logger.info(f"\n🎊 ULTRA-OPTIMIZED NODE INJECTION COMPLETE!")
    logger.info(f"📊 Performance Summary:")
    logger.info(f"   • Node types: {successful_nodes}/{len(nodes)} successful")
    logger.info(f"   • Records: {total_processed:,} processed, {total_failed:,} failed")
    logger.info(f"   • Success rate: {overall_success_rate:.1f}%")
    logger.info(f"   • Duration: {total_duration:.2f}s")
    logger.info(f"   • Throughput: {overall_throughput:.0f} nodes/second")
    
    return final_stats
