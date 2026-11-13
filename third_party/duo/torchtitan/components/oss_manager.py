"""
OSS File Manager Library

A high-performance multi-threaded file manager for Alibaba Cloud Object Storage Service (OSS).
Supports efficient upload/download of large files with automatic fallback mechanisms.

Features:
- Multi-threaded multipart upload/download
- Automatic fallback to single-threaded operations
- Configurable chunk size and thread count
- Verbose logging control
- PyTorch model save/load integration
- Context manager support

Author: Your Name
Version: 1.0.0
"""

import torch
import io
import os
import oss2
import threading
import concurrent.futures
from typing import Optional, Any, Dict
from contextlib import contextmanager
import math


FILE_APPEND_NEXT_POSITION_CACHES = dict()


class OSSFile:
    """
    OSS file class with multi-threaded upload/download support
    
    This class provides a file-like interface for OSS objects with automatic
    multi-threaded operations for large files.
    """
    
    def __init__(self, bucket, oss_path: str, mode: str = 'wb', 
                 chunk_size: int = 10*1024*1024, max_workers: int = 4, verbose: bool = True):
        """
        Initialize OSS file object
        
        Args:
            bucket: OSS bucket instance
            oss_path: Path to the OSS object
            mode: File mode ('wb', 'rb', 'ab')
            chunk_size: Size of each chunk for multipart operations (bytes)
            max_workers: Maximum number of threads for concurrent operations
            verbose: Enable detailed logging
        """
        self.bucket = bucket
        self.oss_path = oss_path
        self.mode = mode
        self.buffer = io.BytesIO()
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.verbose = verbose
        self._closed = False
        self._lock = threading.Lock()

    def _log(self, message: str) -> None:
        """Log message if verbose is enabled"""
        if self.verbose:
            print(message)

    def write(self, data: bytes) -> int:
        """Write data to buffer"""
        if self._closed:
            raise ValueError("File is closed")
        with self._lock:
            return self.buffer.write(data)

    def read(self, size: int = -1) -> bytes:
        """Read data from buffer"""
        if self._closed:
            raise ValueError("File is closed")
        return self.buffer.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position in buffer"""
        return self.buffer.seek(offset, whence)

    def tell(self) -> int:
        """Get current position in buffer"""
        return self.buffer.tell()

    def flush(self) -> None:
        """Flush buffer (no-op for OSS)"""
        pass

    def _upload_chunk(self, upload_id: str, part_number: int, data: bytes) -> oss2.models.PartInfo:
        """Upload a single chunk"""
        chunk_size_mb = len(data) / (1024 * 1024)
        self._log(f"Uploading part {part_number}, size: {chunk_size_mb:.2f}MB")
        
        try:
            result = self.bucket.upload_part(self.oss_path, upload_id, part_number, data)
            self._log(f"Part {part_number} uploaded successfully, ETag: {result.etag}")
            return oss2.models.PartInfo(part_number, result.etag)
        except Exception as e:
            self._log(f"Part {part_number} upload failed: {e}")
            raise

    def _multipart_upload(self, data: bytes) -> bool:
        """Multi-threaded multipart upload with fallback to normal upload"""
        total_size_mb = len(data) / (1024 * 1024)
        self._log(f"Preparing to upload file to {self.oss_path}, total size: {total_size_mb:.2f}MB")
        
        if len(data) <= self.chunk_size:
            # Small file - use direct upload
            self._log("File is small, using direct upload")
            try:
                result = self.bucket.put_object(self.oss_path, data)
                self._log(f"Direct upload completed, status code: {result.status}")
                return result.status == 200
            except Exception as e:
                self._log(f"Direct upload failed: {e}")
                return False
        
        # Large file - try multipart upload
        chunk_size_mb = self.chunk_size / (1024 * 1024)
        num_parts = math.ceil(len(data) / self.chunk_size)
        self._log(f"File is large, using multipart upload, chunk size: {chunk_size_mb:.2f}MB, total {num_parts} parts")
        
        try:
            self._log("Initializing multipart upload...")
            upload_result = self.bucket.init_multipart_upload(self.oss_path)
            upload_id = upload_result.upload_id
            self._log(f"Multipart upload ID: {upload_id}")
            
            # Split into chunks
            total_size = len(data)
            
            # Multi-threaded upload
            parts = []
            self._log(f"Starting concurrent upload with {self.max_workers} threads...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for i in range(num_parts):
                    start = i * self.chunk_size
                    end = min(start + self.chunk_size, total_size)
                    chunk_data = data[start:end]
                    future = executor.submit(self._upload_chunk, upload_id, i + 1, chunk_data)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    parts.append(future.result())
            
            # Complete upload
            self._log("All parts uploaded, merging...")
            parts.sort(key=lambda x: x.part_number)
            self.bucket.complete_multipart_upload(self.oss_path, upload_id, parts)
            self._log("Multipart upload completed successfully!")
            return True
            
        except Exception as e:
            self._log(f"Multipart upload failed, falling back to direct upload: {e}")
            # Fallback to direct upload
            try:
                self._log("Trying direct upload...")
                result = self.bucket.put_object(self.oss_path, data)
                self._log(f"Direct upload completed, status code: {result.status}")
                return result.status == 200
            except Exception as e2:
                self._log(f"Direct upload also failed: {e2}")
                return False

    def _append_upload(self, data: bytes) -> bool:
        """Append upload"""
        total_size_mb = len(data) / (1024 * 1024)
        self._log(f"Preparing to append upload file to {self.oss_path}, total size: {total_size_mb:.2f}MB")
        try:
            if self.oss_path in FILE_APPEND_NEXT_POSITION_CACHES:
                position = FILE_APPEND_NEXT_POSITION_CACHES[self.oss_path]
            else:
                try:
                    meta = self.bucket.head_object(self.oss_path)
                    position = meta.content_length
                except Exception:
                    position = 0
            result = self.bucket.append_object(self.oss_path, position=position, data=data)
            FILE_APPEND_NEXT_POSITION_CACHES[self.oss_path] = result.next_position
            self._log(f"Append upload completed, status code: {result.status}")
            return result.status == 200
        except Exception as e:
            self._log(f"Append upload failed: {e}")
            return False

    def _download_chunk(self, start: int, end: int) -> tuple:
        """Download a single chunk"""
        chunk_size_mb = (end - start) / (1024 * 1024)
        self._log(f"Downloading range {start}-{end-1}, size: {chunk_size_mb:.2f}MB")
        
        try:
            headers = {'Range': f'bytes={start}-{end-1}'}
            result = self.bucket.get_object(self.oss_path, headers=headers)
            data = result.read()
            self._log(f"Downloaded range {start}-{end-1} successfully")
            return start, data
        except Exception as e:
            self._log(f"Download range {start}-{end-1} failed: {e}")
            raise

    def _multipart_download(self) -> bytes:
        """Multi-threaded multipart download with fallback to normal download"""
        try:
            self._log(f"Preparing to download file from {self.oss_path}...")
            # Get file size
            meta = self.bucket.head_object(self.oss_path)
            total_size = meta.content_length
            total_size_mb = total_size / (1024 * 1024)
            self._log(f"File total size: {total_size_mb:.2f}MB")
            
            if total_size <= self.chunk_size:
                # Small file - direct download
                self._log("File is small, using direct download")
                result = self.bucket.get_object(self.oss_path)
                data = result.read()
                self._log("Direct download completed")
                return data
            
            # Large file - multipart download
            chunk_size_mb = self.chunk_size / (1024 * 1024)
            num_parts = math.ceil(total_size / self.chunk_size)
            self._log(f"File is large, using multipart download, chunk size: {chunk_size_mb:.2f}MB, total {num_parts} parts")
            
            chunks = {}
            
            self._log(f"Starting concurrent download with {self.max_workers} threads...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for i in range(num_parts):
                    start = i * self.chunk_size
                    end = min(start + self.chunk_size, total_size)
                    future = executor.submit(self._download_chunk, start, end)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    start, chunk_data = future.result()
                    chunks[start] = chunk_data
            
            # Merge chunks
            self._log("All parts downloaded, merging...")
            result_data = b''
            for i in range(num_parts):
                start = i * self.chunk_size
                result_data += chunks[start]
            
            self._log("Multipart download completed successfully!")
            return result_data
            
        except Exception as e:
            self._log(f"Multipart download failed, falling back to direct download: {e}")
            # Fallback to direct download
            self._log("Trying direct download...")
            result = self.bucket.get_object(self.oss_path)
            data = result.read()
            self._log("Direct download completed")
            return data

    def close(self) -> None:
        """Close file and perform upload if in write mode"""
        if self._closed:
            return
        
        if 'w' in self.mode or 'a' in self.mode:
            self.buffer.seek(0)
            data = self.buffer.getvalue()
            if 'w' in self.mode:
                self._multipart_upload(data)
            else:
                self._append_upload(data)
            del data
        
        self.buffer.close()
        self.buffer = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        import gc
        gc.collect()

    @property
    def closed(self) -> bool:
        """Check if file is closed"""
        return self._closed


class OSSFileManager:
    """
    OSS file manager with multi-threaded upload/download support
    
    Main interface for OSS operations with automatic optimization for large files.
    Provides PyTorch model integration and flexible configuration options.
    """
    
    def __init__(self, access_key_id: str, access_key_secret: str, endpoint: str, bucket_name: str, 
                 chunk_size: int = 10*1024*1024, max_workers: int = 4, verbose: bool = True):
        """
        Initialize OSS connection
        
        Args:
            access_key_id: Alibaba Cloud AccessKey ID
            access_key_secret: Alibaba Cloud AccessKey Secret  
            endpoint: OSS endpoint (e.g., 'oss-cn-beijing.aliyuncs.com')
            bucket_name: OSS bucket name
            chunk_size: Chunk size for multipart operations (bytes), default 10MB
            max_workers: Maximum thread count for concurrent operations, default 4
            verbose: Enable detailed logging, default True
        """
        auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(auth, endpoint, bucket_name)
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Log message if verbose is enabled"""
        if self.verbose:
            print(message)

    @contextmanager
    def open(self, oss_path: str, mode: str = 'wb'):
        """
        Open OSS file with automatic multi-threaded operations
        
        Args:
            oss_path: OSS file path
            mode: File mode ('wb', 'rb', 'ab')
            
        Yields:
            OSSFile: File-like object for OSS operations
        """
        oss_file = OSSFile(self.bucket, oss_path, mode, self.chunk_size, self.max_workers, self.verbose)
        
        if 'r' in mode:
            # Read mode: multi-threaded download
            try:
                data = oss_file._multipart_download()
                oss_file.buffer.write(data)
                oss_file.buffer.seek(0)
                yield oss_file
            finally:
                oss_file.close()
        else:
            # Write mode
            try:
                yield oss_file
            finally:
                oss_file.close()

    def upload_file_checked(self, local_path, oss_path, *, sha256, retries=3):
        """Upload a disk file without buffering it in RAM; fail on any incomplete write."""
        import time
        size = os.path.getsize(local_path)
        crc = oss2.utils.Crc64()
        with open(local_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                crc(chunk)
        for attempt in range(retries):
            try:
                # The SDK's bucket CRC check remains enabled for each transfer.
                result = self.bucket.put_object_from_file(oss_path, str(local_path),
                    headers={'x-oss-meta-sha256': sha256, 'x-oss-forbid-overwrite': 'true'})
                if result.status != 200:
                    raise IOError('OSS upload did not complete')
                info = self.bucket.head_object(oss_path)
                if info.content_length != size or info.headers.get('x-oss-meta-sha256') != sha256 or info.server_crc != crc.crc:
                    raise IOError('OSS uploaded object differs')
                return dict(bytes=size, sha256=sha256)
            except Exception as error:
                if getattr(error, 'status', None) == 409:
                    info = self.bucket.head_object(oss_path)
                    if info.content_length != size or info.headers.get('x-oss-meta-sha256') != sha256 or info.server_crc != crc.crc:
                        raise ValueError('Refusing to overwrite another OSS feature: ' + oss_path) from error
                    return dict(bytes=size, sha256=sha256)
                if attempt + 1 == retries:
                    raise
                time.sleep(min(2 ** attempt, 8))

    def save_model(self, model: torch.nn.Module, oss_path: str, extra_data: Optional[Dict] = None) -> None:
        """
        Save PyTorch model to OSS with multi-threaded upload
        
        Args:
            model: PyTorch model to save
            oss_path: OSS storage path
            extra_data: Additional data to save with the model
        """
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
        }
        
        if extra_data:
            save_data.update(extra_data)
        
        self._log(f"Multi-threaded uploading model to {oss_path}...")
        with self.open(oss_path, 'wb') as f:
            torch.save(save_data, f)
        self._log("Model upload completed!")

    def upload_from_local(self, local_path: str, oss_path: str) -> None:
        """Save a local file/folder to OSS

        Args:
            local_path: Local file/folder path
            oss_path: OSS storage path
        """
        if os.path.isdir(local_path):
            self._log(f"Uploading a local folder from {local_path} to {oss_path}...")
            for folder, _, fnames in os.walk(local_path):
                for fname in fnames:
                    _local_path = os.path.join(folder, fname)
                    rela_path = os.path.relpath(_local_path, local_path)
                    _oss_path = os.path.join(oss_path, rela_path)
                    self.bucket.put_object_from_file(key=_oss_path, filename=_local_path)
            self._log("Folder upload completed!")
        else:
            self._log(f"Uploading a local file from {local_path} to {oss_path}...")
            self.bucket.put_object_from_file(key=oss_path, filename=local_path)
            self._log("File upload completed!")

    def load_model(self, model: torch.nn.Module, oss_path: str) -> tuple:
        """
        Load PyTorch model from OSS with multi-threaded download
        
        Args:
            model: Model instance to load weights into
            oss_path: OSS storage path
        
        Returns:
            tuple: (model, checkpoint_data)
        """
        self._log(f"Multi-threaded downloading model from {oss_path}...")
        with self.open(oss_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        self._log("Model download completed!")
        return model, checkpoint

    def exists(self, oss_path: str) -> bool:
        """
        Check if OSS object exists
        
        Args:
            oss_path: OSS file path
            
        Returns:
            bool: True if object exists, False otherwise
        """
        try:
            self.bucket.head_object(oss_path)
            return True
        except oss2.exceptions.NoSuchKey:
            try:
                if not oss_path.endswith('/'):
                    oss_path += '/'
                object_list = self.bucket.list_objects_v2(prefix=oss_path).object_list
                return len(object_list) > 0
            except oss2.exceptions.NoSuchKey:
                return False
        except Exception:
            return False

    def get_file_size(self, oss_path: str) -> Optional[int]:
        """
        Get file size in bytes
        
        Args:
            oss_path: OSS file path
            
        Returns:
            int: File size in bytes, None if file doesn't exist
        """
        try:
            meta = self.bucket.head_object(oss_path)
            return meta.content_length
        except Exception:
            return None

    def delete(self, oss_path: str) -> bool:
        """
        Delete OSS object
        
        Args:
            oss_path: OSS file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.bucket.delete_object(oss_path)
            self._log(f"Deleted {oss_path}")
            return True
        except Exception as e:
            self._log(f"Failed to delete {oss_path}: {e}")
            return False

    def make_directory(self, oss_directory: str) -> bool:
        """
        Make OSS directory

        Args:
            oss_directory: OSS directory

        Returns:
            bool: True if successful, False otherwise
        """
        if not oss_directory.endswith('/'):
            oss_directory += '/'
        try:
            self.bucket.put_object(oss_directory, '')
            self._log(f"Made directory {oss_directory}")
            return True
        except Exception as e:
            self._log(f"Failed to make directory {oss_directory}: {e}")
            return False

    def list_objects(self, prefix: str = '', max_keys: int = 100) -> list:
        """
        List objects with given prefix
        
        Args:
            prefix: Object key prefix
            max_keys: Maximum number of objects to return
            
        Returns:
            list: List of object keys
        """
        try:
            result = self.bucket.list_objects_v2(prefix=prefix, max_keys=max_keys)
            return [obj.key for obj in result.object_list]
        except Exception as e:
            self._log(f"Failed to list objects: {e}")
            return []

    def list_prefixs(self, prefix: str = '', max_keys: int = 100) -> list:
        """
        List prefixs with given prefix
        
        Args:
            prefix: Prefix key prefix
            max_keys: Maximum number of prefixs to return
            
        Returns:
            list: List of prefix keys
        """
        try:
            result = self.bucket.list_objects_v2(prefix=prefix, max_keys=max_keys, delimiter='/')
            return result.prefix_list
        except Exception as e:
            self._log(f"Failed to list prefixs: {e}")
            return []

    def set_config(self, chunk_size: Optional[int] = None, max_workers: Optional[int] = None, 
                   verbose: Optional[bool] = None) -> None:
        """
        Update configuration settings
        
        Args:
            chunk_size: New chunk size for multipart operations
            max_workers: New maximum thread count
            verbose: Enable/disable detailed logging
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
            self._log(f"Updated chunk size to {chunk_size / (1024*1024):.1f}MB")
        if max_workers is not None:
            self.max_workers = max_workers
            self._log(f"Updated max workers to {max_workers}")
        if verbose is not None:
            self.verbose = verbose
            self._log(f"Updated verbose to {verbose}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration
        
        Returns:
            dict: Current configuration settings
        """
        return {
            'chunk_size': self.chunk_size,
            'chunk_size_mb': self.chunk_size / (1024*1024),
            'max_workers': self.max_workers,
            'verbose': self.verbose,
            'bucket_name': self.bucket.bucket_name,
            'endpoint': self.bucket.endpoint
        }


def create_oss_manager(access_key_id: str, access_key_secret: str, endpoint: str, bucket_name: str, 
                      chunk_size: int = 10*1024*1024, max_workers: int = 4, verbose: bool = True, **kwargs) -> OSSFileManager:
    """
    Create OSS file manager with specified configuration
    
    Args:
        access_key_id: Alibaba Cloud AccessKey ID
        access_key_secret: Alibaba Cloud AccessKey Secret
        endpoint: OSS endpoint
        bucket_name: OSS bucket name
        chunk_size: Chunk size for multipart operations (bytes)
        max_workers: Maximum thread count
        verbose: Enable detailed logging
    
    Returns:
        OSSFileManager: Configured OSS manager instance
    """
    return OSSFileManager(access_key_id, access_key_secret, endpoint, bucket_name, 
                         chunk_size, max_workers, verbose)


# Version information
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Export main classes and functions
__all__ = [
    'OSSFileManager',
    'OSSFile', 
    'create_oss_manager',
    '__version__'
]