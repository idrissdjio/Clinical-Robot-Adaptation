#!/usr/bin/env python3
"""
Backup and Restore Utilities
Backup and restore utilities for clinical robotics data and configurations.
"""

import os
import shutil
import gzip
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import hashlib
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class BackupInfo:
    """Information about a backup."""
    backup_id: str
    timestamp: datetime
    size_bytes: int
    checksum: str
    files: List[str]
    metadata: Dict[str, Any]

class BackupManager:
    """Backup manager for clinical robotics data."""
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, source_paths: List[str], 
                     backup_name: Optional[str] = None,
                     compress: bool = True,
                     include_metadata: bool = True) -> BackupInfo:
        """Create backup of specified paths."""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        files_backed = []
        total_size = 0
        
        for source_path in source_paths:
            source = Path(source_path)
            
            if not source.exists():
                logger.warning(f"Source path does not exist: {source_path}")
                continue
            
            if source.is_file():
                dest = backup_path / source.name
                shutil.copy2(source, dest)
                files_backed.append(str(dest))
                total_size += dest.stat().st_size
            elif source.is_dir():
                dest = backup_path / source.name
                shutil.copytree(source, dest)
                files_backed.extend(str(p) for p in dest.rglob('*') if p.is_file())
                total_size += sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())
        
        # Calculate checksum
        checksum = self._calculate_checksum(backup_path)
        
        # Create backup info
        backup_info = BackupInfo(
            backup_id=backup_name,
            timestamp=datetime.now(),
            size_bytes=total_size,
            checksum=checksum,
            files=files_backed,
            metadata={'compressed': compress} if include_metadata else {}
        )
        
        # Save backup info
        self._save_backup_info(backup_info, backup_path)
        
        # Compress if requested
        if compress:
            self._compress_backup(backup_path)
        
        logger.info(f"Backup created: {backup_name} ({total_size} bytes)")
        return backup_info
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of backup."""
        sha256_hash = hashlib.sha256()
        
        if path.is_file():
            with open(path, 'rb') as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        else:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        for byte_block in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _save_backup_info(self, backup_info: BackupInfo, backup_path: Path):
        """Save backup information to file."""
        info_file = backup_path / "backup_info.json"
        
        info_data = {
            'backup_id': backup_info.backup_id,
            'timestamp': backup_info.timestamp.isoformat(),
            'size_bytes': backup_info.size_bytes,
            'checksum': backup_info.checksum,
            'files': backup_info.files,
            'metadata': backup_info.metadata
        }
        
        with open(info_file, 'w') as f:
            json.dump(info_data, f, indent=2)
    
    def _compress_backup(self, backup_path: Path):
        """Compress backup directory."""
        archive_path = backup_path.parent / f"{backup_path.name}.tar.gz"
        
        shutil.make_archive(
            str(backup_path.parent / backup_path.name),
            'gztar',
            str(backup_path.parent),
            backup_path.name
        )
        
        # Remove uncompressed directory
        shutil.rmtree(backup_path)
        
        logger.info(f"Backup compressed: {archive_path}")
    
    def restore_backup(self, backup_id: str, destination: str,
                     verify_checksum: bool = True) -> bool:
        """Restore backup from archive."""
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        
        if not backup_path.exists():
            backup_path = self.backup_dir / backup_id
        
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        dest_path = Path(destination)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Extract backup
        if backup_path.suffix == '.gz':
            shutil.unpack_archive(str(backup_path), str(self.backup_dir))
            extracted_path = self.backup_dir / backup_path.stem
        else:
            extracted_path = backup_path
        
        # Verify checksum if requested
        if verify_checksum:
            backup_info = self._load_backup_info(extracted_path)
            current_checksum = self._calculate_checksum(extracted_path)
            
            if backup_info.checksum != current_checksum:
                logger.error("Checksum verification failed")
                return False
        
        # Restore files
        backup_info = self._load_backup_info(extracted_path)
        
        for file_path in backup_info.files:
            src = Path(file_path)
            rel_path = src.relative_to(extracted_path)
            dest = dest_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        
        logger.info(f"Backup restored to: {destination}")
        return True
    
    def _load_backup_info(self, backup_path: Path) -> BackupInfo:
        """Load backup information from file."""
        info_file = backup_path / "backup_info.json"
        
        with open(info_file, 'r') as f:
            info_data = json.load(f)
        
        return BackupInfo(
            backup_id=info_data['backup_id'],
            timestamp=datetime.fromisoformat(info_data['timestamp']),
            size_bytes=info_data['size_bytes'],
            checksum=info_data['checksum'],
            files=info_data['files'],
            metadata=info_data.get('metadata', {})
        )
    
    def list_backups(self) -> List[BackupInfo]:
        """List all available backups."""
        backups = []
        
        for item in self.backup_dir.iterdir():
            if item.suffix == '.gz':
                # Extract info from compressed backup
                info_file = self.backup_dir / item.stem / "backup_info.json"
                if info_file.exists():
                    backup_info = self._load_backup_info(self.backup_dir / item.stem)
                    backups.append(backup_info)
            elif item.is_dir():
                info_file = item / "backup_info.json"
                if info_file.exists():
                    backup_info = self._load_backup_info(item)
                    backups.append(backup_info)
        
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        
        if not backup_path.exists():
            backup_path = self.backup_dir / backup_id
        
        if backup_path.exists():
            if backup_path.is_dir():
                shutil.rmtree(backup_path)
            else:
                backup_path.unlink()
            
            logger.info(f"Backup deleted: {backup_id}")
            return True
        
        return False

class DatabaseBackup:
    """Database-specific backup utilities."""
    
    def __init__(self, db_path: str, backup_dir: str = "./backups"):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def backup_database(self, backup_name: Optional[str] = None) -> str:
        """Backup SQLite database."""
        if not backup_name:
            backup_name = f"db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        backup_path = self.backup_dir / backup_name
        
        # Copy database file
        shutil.copy2(self.db_path, backup_path)
        
        logger.info(f"Database backed up to: {backup_path}")
        return str(backup_path)
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup."""
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        # Verify backup is valid SQLite database
        try:
            conn = sqlite3.connect(str(backup_file))
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            conn.close()
        except sqlite3.DatabaseError:
            logger.error("Invalid SQLite database file")
            return False
        
        # Restore database
        shutil.copy2(backup_file, self.db_path)
        
        logger.info(f"Database restored from: {backup_path}")
        return True

def main():
    """Main function for backup/restore utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup and Restore Utilities')
    parser.add_argument('--action', type=str, required=True,
                       choices=['backup', 'restore', 'list', 'delete'],
                       help='Action to perform')
    parser.add_argument('--source', type=str, nargs='+', help='Source paths to backup')
    parser.add_argument('--destination', type=str, help='Destination for restore')
    parser.add_argument('--backup-id', type=str, help='Backup ID')
    parser.add_argument('--backup-dir', type=str, default='./backups',
                       help='Backup directory')
    
    args = parser.parse_args()
    
    manager = BackupManager(args.backup_dir)
    
    if args.action == 'backup':
        if not args.source:
            print("Error: --source required for backup")
            return
        
        backup_info = manager.create_backup(args.source)
        print(f"Backup created: {backup_info.backup_id}")
    
    elif args.action == 'restore':
        if not args.backup_id or not args.destination:
            print("Error: --backup-id and --destination required for restore")
            return
        
        success = manager.restore_backup(args.backup_id, args.destination)
        if success:
            print("Restore completed successfully")
        else:
            print("Restore failed")
    
    elif args.action == 'list':
        backups = manager.list_backups()
        print(f"Available backups ({len(backups)}):")
        for backup in backups:
            print(f"  - {backup.backup_id} ({backup.timestamp}) - {backup.size_bytes} bytes")
    
    elif args.action == 'delete':
        if not args.backup_id:
            print("Error: --backup-id required for delete")
            return
        
        success = manager.delete_backup(args.backup_id)
        if success:
            print(f"Backup deleted: {args.backup_id}")
        else:
            print("Delete failed")

if __name__ == '__main__':
    main()
