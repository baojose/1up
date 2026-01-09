"""
Cleanup script for old images and crops.
Removes scenes older than N days, optionally keeping only useful crops.
"""
import yaml
import logging
import argparse
from pathlib import Path

from storage import ImageStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Clean up old 1UP images and crops")
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Keep scenes from last N days (default: 7)'
    )
    parser.add_argument(
        '--keep-all',
        action='store_true',
        help='Keep all crops, not just useful ones'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize storage
    storage = ImageStorage(config)
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No files will be deleted")
        logger.info(f"   Would clean scenes older than {args.days} days")
        logger.info(f"   Keep useful only: {not args.keep_all}")
        # TODO: Implement dry-run logic
        logger.info("   (Dry-run not fully implemented, use without --dry-run to clean)")
    else:
        # Perform cleanup
        stats = storage.cleanup_old_scenes(
            keep_days=args.days,
            keep_useful_only=not args.keep_all
        )
        
        logger.info(f"\n✅ Cleanup complete!")
        logger.info(f"   Freed {stats['bytes_freed'] / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

