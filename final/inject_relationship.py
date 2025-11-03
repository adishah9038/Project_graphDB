from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
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
import random
from functools import wraps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BatchMetrics:
    """Track performance metrics for adaptive optimization"""
    relationship: str
    batch_size: int
    records: int
    duration: float
    success_rate: float
    contention_level: str
    retry_count: int = 0

@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True  # Add random jitter to prevent thundering herd

class RetryableException(Exception):
    """Exception that should trigger a retry"""
    pass

class NonRetryableException(Exception):
    """Exception that should not trigger a retry"""
    pass

def retry_with_exponential_backoff(retry_config: RetryConfig):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except NonRetryableException:
                    # Don't retry these exceptions
                    raise
                except Exception as e:
                    last_exception = e
                    
                    if attempt >= retry_config.max_retries:
                        logger.error(f"Max retries ({retry_config.max_retries}) exceeded for {func.__name__}")
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        retry_config.base_delay * (retry_config.backoff_multiplier ** attempt),
                        retry_config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if retry_config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                                 f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator

class AdaptiveBatchOptimizer:
    """Dynamically optimize batch sizes based on performance feedback with retry awareness"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.optimal_sizes = {}
        self.failure_patterns = defaultdict(list)
        
    def record_batch_performance(self, metrics: BatchMetrics):
        """Record batch performance for learning"""
        self.metrics_history[metrics.relationship].append(metrics)
        
        # Track failure patterns
        if metrics.success_rate < 0.8:  # Consider < 80% success as problematic
            self.failure_patterns[metrics.relationship].append({
                'batch_size': metrics.batch_size,
                'success_rate': metrics.success_rate,
                'retry_count': metrics.retry_count
            })
        
    def get_optimal_batch_size(self, relationship: str, default_size: int = 500) -> int:
        """Calculate optimal batch size based on historical performance and failures"""
        if relationship not in self.metrics_history:
            return default_size
            
        history = self.metrics_history[relationship][-15:]  # Last 15 batches
        if len(history) < 3:
            return default_size
        
        # Penalize batch sizes that frequently fail
        size_scores = defaultdict(list)
        for metric in history:
            # Score combines throughput with success rate and retry penalty
            throughput = metric.records / metric.duration if metric.duration > 0 else 0
            retry_penalty = 1.0 / (1.0 + metric.retry_count * 0.5)  # Reduce score for retries
            score = throughput * metric.success_rate * retry_penalty
            size_scores[metric.batch_size].append(score)
        
        # Find batch size with best average score
        best_size = default_size
        best_score = 0
        
        for size, scores in size_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_size = size
        
        # Apply safety constraints based on failure patterns
        failures = self.failure_patterns.get(relationship, [])
        if failures:
            # If we've had failures with large batches, cap the size
            failed_sizes = [f['batch_size'] for f in failures if f['success_rate'] < 0.5]
            if failed_sizes:
                max_safe_size = min(failed_sizes) if failed_sizes else best_size
                best_size = min(best_size, max_safe_size // 2)
        
        # Ensure minimum viable size
        return max(50, min(best_size, 2000))

class IntelligentChunker:
    """Advanced data chunking with contention awareness and retry optimization"""
    
    def __init__(self, optimizer: AdaptiveBatchOptimizer):
        self.optimizer = optimizer
        
    def analyze_contention_pattern(self, df: pd.DataFrame, source_col: str, target_col: str) -> Dict:
        """Deep analysis of data contention patterns"""
        
        # Count degrees
        source_counts = df[source_col].value_counts()
        target_counts = df[target_col].value_counts()
        
        # Statistical analysis
        source_stats = {
            'mean': source_counts.mean(),
            'std': source_counts.std(),
            'max': source_counts.max(),
            'p95': source_counts.quantile(0.95),
            'high_contention': source_counts[source_counts > source_counts.quantile(0.90)].index.tolist()
        }
        
        target_stats = {
            'mean': target_counts.mean(),
            'std': target_counts.std(),
            'max': target_counts.max(),
            'p95': target_counts.quantile(0.95),
            'high_contention': target_counts[target_counts > target_counts.quantile(0.90)].index.tolist()
        }
        
        # Determine contention level (more conservative thresholds)
        source_contention = "HIGH" if source_stats['max'] > source_stats['mean'] * 3 else "MEDIUM" if source_stats['max'] > source_stats['mean'] * 1.5 else "LOW"
        target_contention = "HIGH" if target_stats['max'] > target_stats['mean'] * 3 else "MEDIUM" if target_stats['max'] > target_stats['mean'] * 1.5 else "LOW"
        
        return {
            'source_stats': source_stats,
            'target_stats': target_stats,
            'source_contention': source_contention,
            'target_contention': target_contention,
            'overall_contention': max(source_contention, target_contention, key=lambda x: {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}[x])
        }
    
    def create_smart_chunks(self, df: pd.DataFrame, source_col: str, target_col: str, 
                           relationship_name: str, base_batch_size: int = 500) -> List[pd.DataFrame]:
        """Create optimized chunks based on contention analysis and historical performance"""
        
        analysis = self.analyze_contention_pattern(df, source_col, target_col)
        optimal_batch_size = self.optimizer.get_optimal_batch_size(relationship_name, base_batch_size)
        
        # Reduce batch size for high contention scenarios
        if analysis['overall_contention'] == 'HIGH':
            optimal_batch_size = min(optimal_batch_size, 200)
        elif analysis['overall_contention'] == 'MEDIUM':
            optimal_batch_size = min(optimal_batch_size, 500)
        
        logger.info(f"Chunking {relationship_name}: {len(df)} records, contention={analysis['overall_contention']}, batch_size={optimal_batch_size}")
        
        # Choose strategy based on contention level and data size
        if len(df) < 1000:
            return self._simple_chunk(df, min(optimal_batch_size, 100))
        elif analysis['overall_contention'] == 'LOW':
            return self._size_based_chunk(df, optimal_batch_size)
        elif analysis['overall_contention'] == 'MEDIUM':
            return self._balanced_chunk(df, source_col, target_col, analysis, optimal_batch_size)
        else:  # HIGH contention
            return self._contention_aware_chunk(df, source_col, target_col, analysis, optimal_batch_size)
    
    def _simple_chunk(self, df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
        """Simple size-based chunking for small datasets"""
        return [df.iloc[i:i+batch_size].copy() for i in range(0, len(df), batch_size)]
    
    def _size_based_chunk(self, df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
        """Hash-based distribution for low contention"""
        num_chunks = max(1, len(df) // batch_size)
        df_copy = df.copy()
        
        # Create hash-based distribution
        df_copy['chunk_id'] = df_copy.apply(lambda row: hash(str(row.values)) % num_chunks, axis=1)
        
        chunks = []
        for chunk_id in range(num_chunks):
            chunk = df_copy[df_copy['chunk_id'] == chunk_id].drop('chunk_id', axis=1)
            if not chunk.empty:
                chunks.append(chunk)
                
        return chunks
    
    def _balanced_chunk(self, df: pd.DataFrame, source_col: str, target_col: str, 
                       analysis: Dict, batch_size: int) -> List[pd.DataFrame]:
        """Balanced chunking for medium contention"""
        
        # Separate high and low contention records
        high_source = set(analysis['source_stats']['high_contention'])
        high_target = set(analysis['target_stats']['high_contention'])
        
        high_contention_mask = (df[source_col].isin(high_source)) | (df[target_col].isin(high_target))
        
        high_df = df[high_contention_mask].copy()
        low_df = df[~high_contention_mask].copy()
        
        chunks = []
        
        # Process high contention records with smaller batches
        if not high_df.empty:
            small_batch_size = max(50, batch_size // 8)  # Much smaller for high contention
            chunks.extend(self._source_isolated_chunk(high_df, source_col, small_batch_size))
        
        # Process low contention records with normal batches
        if not low_df.empty:
            chunks.extend(self._size_based_chunk(low_df, batch_size))
            
        return chunks
    
    def _contention_aware_chunk(self, df: pd.DataFrame, source_col: str, target_col: str,
                              analysis: Dict, batch_size: int) -> List[pd.DataFrame]:
        """Advanced chunking for high contention scenarios - minimize lock conflicts"""
        
        chunks = []
        processed_indices = set()
        
        # Strategy 1: Isolate super high-contention nodes (top 5)
        super_high_sources = df[source_col].value_counts().head(5).index.tolist()
        super_high_targets = df[target_col].value_counts().head(5).index.tolist()
        
        # Process each super high-contention source individually
        for source in super_high_sources:
            source_mask = (df[source_col] == source)
            if source_mask.sum() > 0:
                source_df = df[source_mask].copy()
                processed_indices.update(source_df.index)
                
                # Split large groups into very small batches to reduce lock time
                micro_batch_size = max(25, batch_size // 20)
                for i in range(0, len(source_df), micro_batch_size):
                    chunk = source_df.iloc[i:i+micro_batch_size]
                    chunks.append(chunk)
        
        # Process remaining high-contention targets
        for target in super_high_targets:
            if target in super_high_sources:  # Avoid double processing
                continue
                
            target_mask = (df[target_col] == target) & (~df.index.isin(processed_indices))
            if target_mask.sum() > 0:
                target_df = df[target_mask].copy()
                processed_indices.update(target_df.index)
                
                micro_batch_size = max(25, batch_size // 20)
                for i in range(0, len(target_df), micro_batch_size):
                    chunk = target_df.iloc[i:i+micro_batch_size]
                    chunks.append(chunk)
        
        # Process remaining records with source isolation
        remaining_df = df[~df.index.isin(processed_indices)]
        if not remaining_df.empty:
            chunks.extend(self._source_isolated_chunk(remaining_df, source_col, batch_size // 4))
            
        return chunks
    
    def _source_isolated_chunk(self, df: pd.DataFrame, source_col: str, batch_size: int) -> List[pd.DataFrame]:
        """Source-isolated chunking to minimize lock contention"""
        
        chunks = []
        source_groups = df.groupby(source_col)
        current_chunk_data = []
        current_chunk_size = 0
        
        for source_id, group in source_groups:
            group_size = len(group)
            
            # If current chunk + new group exceeds batch size, finalize current chunk
            if current_chunk_size + group_size > batch_size and current_chunk_data:
                chunks.append(pd.concat(current_chunk_data, ignore_index=True))
                current_chunk_data = []
                current_chunk_size = 0
            
            # If single group is larger than batch size, split it
            if group_size > batch_size:
                for i in range(0, group_size, batch_size):
                    chunk = group.iloc[i:i+batch_size].copy()
                    chunks.append(chunk)
            else:
                current_chunk_data.append(group)
                current_chunk_size += group_size
        
        # Add final chunk if exists
        if current_chunk_data:
            chunks.append(pd.concat(current_chunk_data, ignore_index=True))
            
        return chunks

class OptimizedNeo4jWriter:
    """High-performance Neo4j writer with comprehensive retry mechanisms"""
    
    def __init__(self, driver, max_connections: int = 4):  # Reduced default connections
        self.driver = driver
        self.max_connections = max_connections
        self.connection_pool = queue.Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self.retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            backoff_multiplier=2.0,
            jitter=True
        )
        
        # Test connection first
        self._test_connection()
        
        # Pre-warm connections (fewer to reduce contention)
        self._warm_connections()
    
    def _test_connection(self):
        """Test the database connection and log status"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection OK' as status, datetime() as time")
                record = result.single()
                logger.info(f"Neo4j connection test: {record['status']} at {record['time']}")
                
                # Test node count to verify database state
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()['node_count']
                logger.info(f"Current database has {node_count} nodes")
                
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    def _warm_connections(self):
        """Pre-establish database connections (reduced count)"""
        for _ in range(min(2, self.max_connections)):  # Only 2 pre-warmed connections
            try:
                session = self.driver.session()
                # Test connection
                session.run("RETURN 1 as test").single()
                self.connection_pool.put(session)
            except Exception as e:
                logger.warning(f"Failed to pre-warm connection: {e}")
    
    def get_session(self):
        """Get a session from the pool"""
        try:
            return self.connection_pool.get(timeout=2)  # Longer timeout
        except queue.Empty:
            # Create new session if pool is empty
            return self.driver.session()
    
    def return_session(self, session):
        """Return session to pool"""
        try:
            self.connection_pool.put(session, timeout=0.1)
        except queue.Full:
            # Pool is full, close the session
            session.close()
    
    @retry_with_exponential_backoff(RetryConfig(max_retries=3, base_delay=0.5, max_delay=10.0))
    def _execute_cypher_with_retry(self, session, cypher_query: str, batch_data: List[Dict]) -> Dict:
        """Execute Cypher with retry logic and detailed error handling"""
        try:
            # Set transaction timeout to prevent long-running transactions
            result = session.run(cypher_query, rows=batch_data, timeout=30.0)
            summary = result.consume()
            
            relationships_created = summary.counters.relationships_created
            
            return {
                'success': True,
                'relationships_created': relationships_created,
                'summary': summary
            }
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Classify errors for retry decisions
            if any(keyword in error_str for keyword in [
                'deadlock', 'lock', 'timeout', 'connection', 'transient', 
                'unavailable', 'network', 'temporary'
            ]):
                # These are retryable errors
                logger.warning(f"Retryable error detected: {str(e)}")
                raise RetryableException(f"Retryable database error: {str(e)}")
            else:
                # These are non-retryable errors (syntax, schema, etc.)
                logger.error(f"Non-retryable error: {str(e)}")
                raise NonRetryableException(f"Non-retryable error: {str(e)}")
    
    def execute_optimized_batch(self, cypher_query: str, batch_data: List[Dict], 
                               relationship_name: str) -> Dict:
        """Execute batch with comprehensive retry mechanisms and error handling"""
        
        session = None
        start_time = time.time()
        retry_count = 0
        
        try:
            session = self.get_session()
            
            # Add small random delay to reduce thundering herd
            time.sleep(random.uniform(0.01, 0.1))
            
            # Execute with retry logic
            try:
                result = self._execute_cypher_with_retry(session, cypher_query, batch_data)
                relationships_created = result['relationships_created']
                
                duration = time.time() - start_time
                
                return {
                    'success': True,
                    'committed': relationships_created,
                    'failed': max(0, len(batch_data) - relationships_created),
                    'duration': duration,
                    'records_per_second': relationships_created / duration if duration > 0 else 0,
                    'retry_count': retry_count
                }
                
            except RetryableException as e:
                # This should not happen here due to decorator, but handle it
                retry_count = getattr(e, 'retry_count', 0)
                raise e
                
            except NonRetryableException as e:
                duration = time.time() - start_time
                logger.error(f"Non-retryable error for {relationship_name}: {str(e)}")
                
                return {
                    'success': False,
                    'committed': 0,
                    'failed': len(batch_data),
                    'duration': duration,
                    'error': str(e),
                    'retry_count': retry_count
                }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Batch execution failed for {relationship_name}: {str(e)}")
            
            return {
                'success': False,
                'committed': 0,
                'failed': len(batch_data),
                'duration': duration,
                'error': str(e),
                'retry_count': retry_count
            }
            
        finally:
            if session:
                self.return_session(session)
    
    def _debug_missing_nodes(self, session, batch_data: List[Dict], relationship_name: str):
        """Debug missing nodes to understand why relationships aren't being created"""
        
        if not batch_data:
            return
            
        # Sample a few records to check
        sample_size = min(3, len(batch_data))
        sample_records = batch_data[:sample_size]
        
        logger.info(f"Debugging missing nodes for {relationship_name}:")
        
        for i, record in enumerate(sample_records):
            # Get the keys from the record
            source_key = None
            target_key = None
            
            # Find source and target keys based on common patterns
            for key in record.keys():
                if key.endswith('_id') or key in ['actor_id', 'film_id', 'customer_id', 'staff_id', 
                                                 'store_id', 'category_id', 'language_id', 'country_id',
                                                 'city_id', 'address_id', 'inventory_id', 'rental_id', 'payment_id']:
                    if source_key is None:
                        source_key = key
                        source_value = record[key]
                    else:
                        target_key = key
                        target_value = record[key]
                        break
            
            if source_key and target_key:
                # Check if source node exists
                source_check_query = f"MATCH (n) WHERE n.{source_key} = $value RETURN count(n) as count"
                try:
                    result = session.run(source_check_query, value=source_value)
                    source_count = result.single()['count']
                    logger.info(f"  Record {i+1}: Source {source_key}={source_value} -> {source_count} nodes found")
                except Exception as e:
                    logger.error(f"  Error checking source {source_key}: {e}")
                
                # Check if target node exists  
                target_check_query = f"MATCH (n) WHERE n.{target_key} = $value RETURN count(n) as count"
                try:
                    result = session.run(target_check_query, value=target_value)
                    target_count = result.single()['count']
                    logger.info(f"  Record {i+1}: Target {target_key}={target_value} -> {target_count} nodes found")
                except Exception as e:
                    logger.error(f"  Error checking target {target_key}: {e}")

class UltraOptimizedInjectionSystem:
    """Ultra-optimized injection system with comprehensive retry and anti-locking mechanisms"""
    
    def __init__(self, driver, output_dir: str, nodes_list: List, initial_batch_size: int = 300, max_workers: int = None):
        self.driver = driver
        self.output_dir = output_dir
        self.initial_batch_size = initial_batch_size
        
        # Create node lookup for key mapping
        self.node_lookup = {node.name: node.key for node in nodes_list}
        logger.info(f"Node lookup: {self.node_lookup}")
        
        # Reduce worker count to minimize contention
        if max_workers is None:
            cpu_count = psutil.cpu_count()
            self.max_workers = min(cpu_count // 2, 3)  # Much more conservative
        else:
            self.max_workers = min(max_workers, 3)  # Cap at 3 workers
            
        self.optimizer = AdaptiveBatchOptimizer()
        self.chunker = IntelligentChunker(self.optimizer)
        self.writer = OptimizedNeo4jWriter(driver, max_connections=self.max_workers)
        
        self.stats = defaultdict(int)
        self.lock = threading.Lock()
        
        logger.info(f"Initialized with {self.max_workers} workers, batch size {initial_batch_size}")
    
    def validate_nodes_exist(self, relationship_schema: Dict) -> Dict:
        """Validate that required nodes exist in the database"""
        
        source_node = relationship_schema['source']
        target_node = relationship_schema['target']
        
        validation_results = {}
        
        try:
            with self.driver.session() as session:
                # Check source nodes
                source_query = f"MATCH (n:{source_node}) RETURN count(n) as count"
                result = session.run(source_query)
                source_count = result.single()['count']
                validation_results['source_count'] = source_count
                
                # Check target nodes  
                target_query = f"MATCH (n:{target_node}) RETURN count(n) as count"
                result = session.run(target_query)
                target_count = result.single()['count']
                validation_results['target_count'] = target_count
                
                logger.info(f"Node validation for {relationship_schema['label']}: "
                           f"{source_node}={source_count}, {target_node}={target_count}")
                
                return validation_results
                
        except Exception as e:
            logger.error(f"Node validation failed: {str(e)}")
            return {'error': str(e)}
    
    def generate_optimized_cypher(self, relationship_schema: Dict) -> str:
        """Generate optimized Cypher query with better performance hints"""
        
        label = relationship_schema['label']
        source_node = relationship_schema['source']
        target_node = relationship_schema['target']
        source_key = relationship_schema['key_s']  # CSV column name
        target_key = relationship_schema['key_t']  # CSV column name
        properties = relationship_schema.get('properties', [])
        
        # Get the actual node key properties
        source_node_key = self.node_lookup.get(source_node, source_key)
        target_node_key = self.node_lookup.get(target_node, target_key)
        
        logger.info(f"Cypher mapping for {label}:")
        logger.info(f"  CSV: {source_key} -> {target_key}")
        logger.info(f"  Nodes: {source_node}.{source_node_key} -> {target_node}.{target_node_key}")
        
        # Build property assignment
        prop_assignments = []
        for prop in properties:
            prop_assignments.append(f"{prop}: row.{prop}")
        
        property_str = "{" + ", ".join(prop_assignments) + "}" if prop_assignments else ""
        
        # Optimized Cypher with hints and proper transaction handling
        cypher = f"""
        UNWIND $rows AS row
        MATCH (s:{source_node}) WHERE s.{source_node_key} = row.{source_key}
        WITH s, row
        MATCH (t:{target_node}) WHERE t.{target_node_key} = row.{target_key}
        WITH s, t, row
        MERGE (s)-[r:{label} {property_str}]->(t)
        RETURN count(r) as relationships_created
        """
        
        return cypher.strip()
    
    def load_and_preprocess_data(self, table_name: str, source_key: str, target_key: str) -> pd.DataFrame:
        """Load and preprocess data with better validation"""
        
        csv_path = f"{self.output_dir}/{table_name}.csv"
        
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return pd.DataFrame()
            
            # Read CSV with optimizations
            df = pd.read_csv(csv_path, engine='c')
            logger.info(f"Loaded {len(df)} records from {csv_path}")
            logger.info(f"CSV columns: {list(df.columns)}")
            
            # Check if required columns exist
            if source_key not in df.columns:
                logger.error(f"Source key '{source_key}' not found in {csv_path}. Available columns: {list(df.columns)}")
                return pd.DataFrame()
                
            if target_key not in df.columns:
                logger.error(f"Target key '{target_key}' not found in {csv_path}. Available columns: {list(df.columns)}")
                return pd.DataFrame()
            
            # Remove rows with null keys
            initial_size = len(df)
            df = df.dropna(subset=[source_key, target_key])
            if len(df) < initial_size:
                logger.info(f"Removed {initial_size - len(df)} rows with null keys")
            
            # Remove duplicates
            df = df.drop_duplicates(subset=[source_key, target_key])
            
            # Convert key columns to ensure proper data types
            if df[source_key].dtype == 'object':
                try:
                    df[source_key] = pd.to_numeric(df[source_key], errors='ignore')
                except:
                    pass
                    
            if df[target_key].dtype == 'object':
                try:
                    df[target_key] = pd.to_numeric(df[target_key], errors='ignore')
                except:
                    pass
            
            # Sort for better locality and reduce lock contention
            df = df.sort_values([source_key, target_key])
            
            logger.info(f"Preprocessed data: {len(df)} records, "
                       f"source_key dtype: {df[source_key].dtype}, "
                       f"target_key dtype: {df[target_key].dtype}")
            
            # Show sample data for debugging
            logger.info(f"Sample data:\n{df.head().to_string()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {csv_path}: {str(e)}")
            return pd.DataFrame()
    
    def process_relationship_optimized(self, relationship_schema: Dict) -> Dict:
        """Process a single relationship with comprehensive retry and anti-locking mechanisms"""
        
        label = relationship_schema['label']
        table_name = relationship_schema['table_name']
        source_key = relationship_schema['key_s']
        target_key = relationship_schema['key_t']
        
        logger.info(f"Processing {label} with enhanced retry mechanisms")
        start_time = time.time()
        
        try:
            # Step 1: Validate that required nodes exist
            validation = self.validate_nodes_exist(relationship_schema)
            if 'error' in validation:
                return {'label': label, 'success': False, 'error': validation['error']}
            
            if validation.get('source_count', 0) == 0:
                logger.error(f"No {relationship_schema['source']} nodes found in database")
                return {'label': label, 'success': False, 'error': f"No {relationship_schema['source']} nodes exist"}
            
            if validation.get('target_count', 0) == 0:
                logger.error(f"No {relationship_schema['target']} nodes found in database")
                return {'label': label, 'success': False, 'error': f"No {relationship_schema['target']} nodes exist"}
            
            # Step 2: Load and preprocess data
            df = self.load_and_preprocess_data(table_name, source_key, target_key)
            
            if df.empty:
                logger.warning(f"No valid data found for {label}")
                return {'label': label, 'success': True, 'committed': 0, 'failed': 0}
            
            # Step 3: Generate optimized Cypher
            cypher_query = self.generate_optimized_cypher(relationship_schema)
            
            # Step 4: Create chunks with conservative sizing for retry scenarios
            conservative_batch_size = min(self.initial_batch_size, 200)  # Start smaller
            chunks = self.chunker.create_smart_chunks(df, source_key, target_key, label, conservative_batch_size)
            
            logger.info(f"Created {len(chunks)} chunks for {label}")
            
            # Step 5: Process chunks with anti-lock mechanisms
            total_committed = 0
            total_failed = 0
            all_metrics = []
            
            # Use minimal concurrency to reduce lock contention
            max_concurrent = min(self.max_workers, 2, len(chunks))  # Maximum 2 concurrent
            
            # Process some chunks sequentially for high contention relationships
            if len(chunks) > 10:  # For large datasets, process some sequentially
                sequential_count = min(3, len(chunks) // 4)  # Process 25% sequentially
                logger.info(f"Processing first {sequential_count} chunks sequentially to reduce contention")
                
                for i in range(sequential_count):
                    chunk_df = chunks[i]
                    batch_data = chunk_df.to_dict('records')
                    result = self.writer.execute_optimized_batch(
                        cypher_query, batch_data, f"{label}_sequential_{i}"
                    )
                    
                    total_committed += result['committed']
                    total_failed += result['failed']
                    
                    # Record metrics for optimization
                    metrics = BatchMetrics(
                        relationship=label,
                        batch_size=len(batch_data),
                        records=result['committed'],
                        duration=result['duration'],
                        success_rate=result['committed'] / len(batch_data) if len(batch_data) > 0 else 0,
                        contention_level="HIGH",
                        retry_count=result.get('retry_count', 0)
                    )
                    all_metrics.append(metrics)
                    self.optimizer.record_batch_performance(metrics)
                    
                    if result['committed'] > 0:
                        logger.info(f"Sequential {label} chunk {i}: {result['committed']} committed")
                    else:
                        logger.warning(f"Sequential {label} chunk {i}: NO relationships created!")
                    
                    # Small delay to reduce contention
                    time.sleep(0.1)
                
                # Process remaining chunks concurrently
                remaining_chunks = chunks[sequential_count:]
            else:
                remaining_chunks = chunks
            
            if remaining_chunks:
                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    
                    future_to_chunk = {}
                    for i, chunk_df in enumerate(remaining_chunks):
                        batch_data = chunk_df.to_dict('records')
                        
                        # Add staggered delays for concurrent execution
                        delay = i * 0.05  # 50ms stagger
                        future = executor.submit(
                            self._delayed_batch_execution,
                            cypher_query, batch_data, f"{label}_concurrent_{i}", delay
                        )
                        future_to_chunk[future] = (i, len(batch_data), batch_data[:2])  # Keep sample for debugging
                    
                    for future in as_completed(future_to_chunk):
                        chunk_id, chunk_size, sample_data = future_to_chunk[future]
                        try:
                            result = future.result()
                            
                            total_committed += result['committed']
                            total_failed += result['failed']
                            
                            # Record metrics for optimization
                            metrics = BatchMetrics(
                                relationship=label,
                                batch_size=chunk_size,
                                records=result['committed'],
                                duration=result['duration'],
                                success_rate=result['committed'] / chunk_size if chunk_size > 0 else 0,
                                contention_level="MEDIUM",
                                retry_count=result.get('retry_count', 0)
                            )
                            all_metrics.append(metrics)
                            self.optimizer.record_batch_performance(metrics)
                            
                            if result['success'] and result['committed'] > 0:
                                logger.info(f"Concurrent {label} chunk {chunk_id}: {result['committed']} committed "
                                          f"in {result['duration']:.2f}s ({result['records_per_second']:.0f} records/s)")
                                if result.get('retry_count', 0) > 0:
                                    logger.info(f"  Required {result['retry_count']} retries")
                            elif result['committed'] == 0:
                                logger.error(f"Concurrent {label} chunk {chunk_id}: NO relationships created! "
                                           f"Sample data: {sample_data}")
                            else:
                                logger.warning(f"Concurrent {label} chunk {chunk_id}: "
                                             f"{result['committed']}/{chunk_size} committed, {result['failed']} failed")
                            
                        except Exception as e:
                            total_failed += chunk_size
                            logger.error(f"Concurrent {label} chunk {chunk_id} exception: {str(e)}")
            
            # Step 6: Final validation and metrics
            actual_rel_count = self._count_actual_relationships(label)
            if actual_rel_count != total_committed:
                logger.warning(f"Mismatch! Expected {total_committed} relationships, but database has {actual_rel_count}")
            
            duration = time.time() - start_time
            success_rate = (total_committed / (total_committed + total_failed)) * 100 if (total_committed + total_failed) > 0 else 0
            overall_throughput = total_committed / duration if duration > 0 else 0
            
            # Calculate retry statistics
            total_retries = sum(m.retry_count for m in all_metrics)
            avg_retries = total_retries / len(all_metrics) if all_metrics else 0
            
            logger.info(f"{label} completed: {total_committed} committed ({actual_rel_count} verified), "
                       f"{total_failed} failed, {success_rate:.1f}% success, {overall_throughput:.0f} records/s, "
                       f"avg retries: {avg_retries:.1f}")
            
            # Cleanup
            del df
            gc.collect()
            
            return {
                'label': label,
                'success': True,
                'committed': total_committed,
                'failed': total_failed,
                'verified_count': actual_rel_count,
                'success_rate': success_rate,
                'duration': duration,
                'throughput': overall_throughput,
                'chunks_processed': len(chunks),
                'total_retries': total_retries,
                'avg_retries': avg_retries
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to process {label}: {str(e)}")
            return {
                'label': label,
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    def _delayed_batch_execution(self, cypher_query: str, batch_data: List[Dict], 
                               relationship_name: str, delay: float = 0) -> Dict:
        """Execute batch with optional delay to reduce thundering herd"""
        if delay > 0:
            time.sleep(delay)
        return self.writer.execute_optimized_batch(cypher_query, batch_data, relationship_name)
    
    def _count_actual_relationships(self, label: str) -> int:
        """Count actual relationships in the database to verify creation"""
        try:
            with self.driver.session() as session:
                result = session.run(f"MATCH ()-[r:{label}]->() RETURN count(r) as count")
                return result.single()['count']
        except Exception as e:
            logger.error(f"Failed to count relationships for {label}: {str(e)}")
            return 0

def execute_ultra_optimized_relationships(relationships: List, driver, output_dir: str, nodes_list: List,
                                        initial_batch_size: int = 300, max_workers: int = None) -> Dict:
    """
    Enhanced relationship execution with comprehensive retry and anti-locking mechanisms
    """
    
    logger.info(f"Starting ENHANCED injection for {len(relationships)} relationships with retry mechanisms")
    start_time = time.time()
    
    # Initialize the system with conservative settings
    system = UltraOptimizedInjectionSystem(driver, output_dir, nodes_list, initial_batch_size, max_workers)
    
    # Pre-flight check: Verify database has nodes
    logger.info("=== PRE-FLIGHT DATABASE CHECK ===")
    try:
        with driver.session() as session:
            # Count total nodes
            result = session.run("MATCH (n) RETURN count(n) as total_nodes")
            total_nodes = result.single()['total_nodes']
            logger.info(f"Database contains {total_nodes} total nodes")
            
            if total_nodes == 0:
                logger.error("DATABASE IS EMPTY! No nodes found. You must load nodes first!")
                return {
                    'error': 'Database is empty - no nodes found',
                    'total_nodes': 0,
                    'success': False
                }
            
            # Count nodes by label
            result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC")
            node_counts = {record['label']: record['count'] for record in result if record['label']}
            logger.info("Node counts by label:")
            for label, count in node_counts.items():
                logger.info(f"  {label}: {count}")
            
            # Count existing relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
            existing_rels = {record['rel_type']: record['count'] for record in result}
            total_existing_rels = sum(existing_rels.values())
            logger.info(f"Database contains {total_existing_rels} existing relationships:")
            for rel_type, count in existing_rels.items():
                logger.info(f"  {rel_type}: {count}")
                
    except Exception as e:
        logger.error(f"Pre-flight check failed: {str(e)}")
        return {'error': f'Pre-flight check failed: {str(e)}', 'success': False}
    
    # Process relationships with enhanced retry logic
    all_results = []
    
    for i, relationship in enumerate(relationships, 1):
        logger.info(f"\nProcessing Relationship {i}/{len(relationships)}: {relationship.label}")
        logger.info(f"   Source: {relationship.source} -> Target: {relationship.target}")
        logger.info(f"   Keys: {relationship.key_s} -> {relationship.key_t}")
        logger.info(f"   Table: {relationship.table_name}")
        
        rel_dict = {
            'label': relationship.label,
            'source': relationship.source,
            'target': relationship.target,
            'key_s': relationship.key_s,
            'key_t': relationship.key_t,
            'properties': relationship.properties,
            'table_name': relationship.table_name
        }
        
        try:
            result = system.process_relationship_optimized(rel_dict)
            all_results.append(result)
            
            # Log immediate result with retry information
            if result.get('success', False):
                committed = result.get('committed', 0)
                verified = result.get('verified_count', 0)
                retries = result.get('total_retries', 0)
                if committed > 0:
                    logger.info(f"{relationship.label}: {committed} relationships created ({verified} verified)")
                    if retries > 0:
                        logger.info(f"  Required {retries} total retries across all batches")
                else:
                    logger.warning(f"{relationship.label}: Processed successfully but NO relationships created")
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"{relationship.label}: Failed - {error}")
            
            # Brief pause between relationships to reduce system stress
            if i < len(relationships):
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"{relationship.label}: Exception - {str(e)}")
            all_results.append({'label': relationship.label, 'success': False, 'error': str(e)})
    
    # Final statistics and verification
    total_duration = time.time() - start_time
    successful_rels = sum(1 for r in all_results if r.get('success', False) and r.get('committed', 0) > 0)
    total_committed = sum(r.get('committed', 0) for r in all_results)
    total_verified = sum(r.get('verified_count', 0) for r in all_results)
    total_failed = sum(r.get('failed', 0) for r in all_results)
    total_retries = sum(r.get('total_retries', 0) for r in all_results)
    
    # Final database verification
    logger.info("\n=== FINAL DATABASE VERIFICATION ===")
    try:
        with driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC")
            final_rels = {record['rel_type']: record['count'] for record in result if record['label']}
            final_total_rels = sum(final_rels.values())
            
            logger.info(f"Final relationship counts:")
            for rel_type, count in final_rels.items():
                logger.info(f"  {rel_type}: {count}")
            logger.info(f"Total relationships in database: {final_total_rels}")
    except Exception as e:
        logger.error(f"Final verification failed: {str(e)}")
        final_total_rels = 0
    
    overall_success_rate = (total_committed / (total_committed + total_failed)) * 100 if (total_committed + total_failed) > 0 else 0
    overall_throughput = total_committed / total_duration if total_duration > 0 else 0
    
    final_stats = {
        'total_relationships': len(relationships),
        'successful_relationships': successful_rels,
        'total_committed': total_committed,
        'total_verified': total_verified,
        'total_failed': total_failed,
        'total_retries': total_retries,
        'overall_success_rate': overall_success_rate,
        'total_duration': total_duration,
        'overall_throughput': overall_throughput,
        'final_db_relationship_count': final_total_rels,
        'detailed_results': all_results
    }
    
    logger.info(f"\nENHANCED EXECUTION COMPLETE!")
    logger.info(f"Performance Summary:")
    logger.info(f"   • Relationships: {successful_rels}/{len(relationships)} successful")
    logger.info(f"   • Records: {total_committed:,} committed, {total_verified:,} verified, {total_failed:,} failed")
    logger.info(f"   • Total retries: {total_retries}")
    logger.info(f"   • Success rate: {overall_success_rate:.1f}%")
    logger.info(f"   • Duration: {total_duration:.2f}s")
    logger.info(f"   • Throughput: {overall_throughput:.0f} records/second")
    logger.info(f"   • Final DB count: {final_total_rels} relationships")
    
    if total_verified == 0:
        logger.error("CRITICAL: NO RELATIONSHIPS VERIFIED IN DATABASE!")
        logger.error("Possible causes:")
        logger.error("   1. Nodes don't exist in database")
        logger.error("   2. Key mismatches between CSV data and node properties")  
        logger.error("   3. Data type mismatches")
        logger.error("   4. Connection/transaction issues")
        logger.error(f"   5. All {total_retries} retries exhausted due to persistent issues")
    
    return final_stats