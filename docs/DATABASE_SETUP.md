# PostgreSQL Database Setup

This guide covers setting up and managing the PostgreSQL database for Bucket Brigade Agent Registry.

## Prerequisites

- PostgreSQL 12 or higher
- Python 3.9+
- Required Python packages (installed via `pip install -e .`)

## Installation

### macOS

```bash
# Install PostgreSQL via Homebrew
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Create database
createdb bucket_brigade
```

### Linux (Ubuntu/Debian)

```bash
# Install PostgreSQL
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database as postgres user
sudo -u postgres createdb bucket_brigade
```

### Docker

```bash
# Run PostgreSQL in Docker
docker run --name bucket-brigade-db \
  -e POSTGRES_DB=bucket_brigade \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -d postgres:15

# Connect to database
docker exec -it bucket-brigade-db psql -U postgres -d bucket_brigade
```

## Configuration

Set the database URL via environment variable:

```bash
# Default (localhost)
export DATABASE_URL="postgresql://localhost:5432/bucket_brigade"

# With credentials
export DATABASE_URL="postgresql://username:password@localhost:5432/bucket_brigade"

# Remote database
export DATABASE_URL="postgresql://username:password@db.example.com:5432/bucket_brigade"
```

### Connection Pool Configuration

Adjust connection pool settings for your workload:

```bash
# Pool size (default: 10)
export DB_POOL_SIZE=20

# Max overflow connections (default: 20)
export DB_MAX_OVERFLOW=40

# Pool timeout in seconds (default: 30)
export DB_POOL_TIMEOUT=60

# Pool recycle time in seconds (default: 3600)
export DB_POOL_RECYCLE=7200

# Enable SQL query logging (default: false)
export DB_ECHO=true
```

## Database Initialization

### Fresh Installation

Initialize the database schema:

```bash
# Initialize database with schema
python -m bucket_brigade.db.migrations.init_db
```

This creates the following tables:
- `agents` - Core agent information
- `submissions` - Submission history
- `agent_metadata` - Extended metadata

### Migration from SQLite

If you have an existing SQLite database:

```bash
# Migrate from SQLite to PostgreSQL
python bucket_brigade/db/migrations/migrate_from_sqlite.py \
  --sqlite-path path/to/db.sqlite
```

Options:
- `--sqlite-path`: Path to SQLite database file (required)
- `--drop-existing`: Drop existing PostgreSQL tables before migration (DANGEROUS!)

Example:

```bash
# Migrate with safety (keeps existing data)
python bucket_brigade/db/migrations/migrate_from_sqlite.py \
  --sqlite-path ./old_database.db

# Fresh migration (drops all existing data)
python bucket_brigade/db/migrations/migrate_from_sqlite.py \
  --sqlite-path ./old_database.db \
  --drop-existing
```

## Database Schema

### Tables

#### agents

Core agent information:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key (auto-increment) |
| name | VARCHAR(255) | Agent display name |
| author | VARCHAR(255) | Agent creator |
| code_path | VARCHAR(512) | Filesystem path to agent code |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update timestamp |
| active | BOOLEAN | Whether agent is active |

Indexes:
- `name` (index)
- `author` (index)
- `active` (index)

#### submissions

Submission history:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key (auto-increment) |
| agent_id | INTEGER | Foreign key to agents.id |
| validation_passed | BOOLEAN | Whether validation succeeded |
| validation_errors | JSON | Array of error messages |
| validation_warnings | JSON | Array of warning messages |
| test_stats | JSON | Test run statistics |
| submitted_at | TIMESTAMP | Submission timestamp |

Indexes:
- `agent_id` (index)
- `submitted_at` (index)

Foreign keys:
- `agent_id` references `agents(id)` ON DELETE CASCADE

#### agent_metadata

Extended metadata:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key (auto-increment) |
| agent_id | INTEGER | Foreign key to agents.id (unique) |
| description | TEXT | Agent description |
| version | VARCHAR(50) | Semantic version |
| tags | JSON | Array of tags/keywords |
| license | VARCHAR(100) | License identifier |
| repository_url | VARCHAR(512) | Repository URL |

Indexes:
- `agent_id` (unique index)

Foreign keys:
- `agent_id` references `agents(id)` ON DELETE CASCADE

## Database Maintenance

### Backup

```bash
# Backup database
pg_dump bucket_brigade > backup.sql

# Backup with compression
pg_dump bucket_brigade | gzip > backup.sql.gz

# Restore from backup
psql bucket_brigade < backup.sql
```

### Monitoring

```bash
# Check connection count
psql bucket_brigade -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'bucket_brigade';"

# Check table sizes
psql bucket_brigade -c "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size FROM pg_tables WHERE schemaname = 'public' ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"

# Check active queries
psql bucket_brigade -c "SELECT pid, query, state FROM pg_stat_activity WHERE datname = 'bucket_brigade';"
```

### Performance Tuning

For high-concurrency workloads:

```sql
-- Analyze tables for query optimization
ANALYZE agents;
ANALYZE submissions;
ANALYZE agent_metadata;

-- Create additional indexes if needed
CREATE INDEX idx_submissions_validation_passed ON submissions(validation_passed);
CREATE INDEX idx_agents_created_at ON agents(created_at DESC);
```

## Troubleshooting

### Connection Issues

```bash
# Test connection
psql -h localhost -p 5432 -U postgres -d bucket_brigade

# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Check PostgreSQL logs
tail -f /usr/local/var/log/postgresql@15.log  # macOS Homebrew
sudo tail -f /var/log/postgresql/postgresql-15-main.log  # Linux
```

### Permission Issues

```sql
-- Grant permissions to user
GRANT ALL PRIVILEGES ON DATABASE bucket_brigade TO your_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;
```

### Migration Failures

If migration fails partway through:

```bash
# Check what tables exist
psql bucket_brigade -c "\dt"

# Drop all tables to start fresh (DANGEROUS!)
psql bucket_brigade -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Re-run initialization
python -m bucket_brigade.db.migrations.init_db
```

## Production Considerations

### Security

1. **Never use default credentials** in production
2. **Enable SSL/TLS** for remote connections
3. **Use connection pooling** (already configured)
4. **Limit database user permissions** to only what's needed
5. **Regular backups** with retention policy

### Scaling

For high-traffic deployments:

1. **Increase connection pool size** via `DB_POOL_SIZE`
2. **Use read replicas** for list/get operations
3. **Add caching layer** (Redis) for frequently accessed data
4. **Monitor query performance** and add indexes as needed
5. **Consider partitioning** submissions table by date

### High Availability

For production deployments:

1. **Primary-Replica setup** with automatic failover
2. **Connection pooling** with PgBouncer
3. **Load balancing** across replicas
4. **Automated backups** with point-in-time recovery
5. **Monitoring and alerting** for database health

## References

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Migration Tool](https://alembic.sqlalchemy.org/)
