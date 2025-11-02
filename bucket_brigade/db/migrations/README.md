# Database Migrations

This directory contains Alembic migrations for the Bucket Brigade database.

## Setup

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Set database URL (optional, defaults to localhost):
   ```bash
   export DATABASE_URL="postgresql://user:password@localhost:5432/bucket_brigade"
   ```

3. Initialize the database:
   ```bash
   python -m bucket_brigade.db.migrations.init_db
   ```

## Migrating from SQLite

If you have an existing SQLite database, run the migration script:

```bash
python bucket_brigade/db/migrations/migrate_from_sqlite.py --sqlite-path path/to/db.sqlite
```

This will:
1. Read all data from SQLite
2. Create PostgreSQL schema
3. Import all data with proper type conversions
4. Verify data integrity

## Creating New Migrations

```bash
alembic revision -m "Description of changes"
```

## Applying Migrations

```bash
alembic upgrade head
```
