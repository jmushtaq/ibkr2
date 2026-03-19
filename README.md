# Setup DB

sudo -u postgres psql
CREATE DATABASE ibkr2_dev;
CREATE USER dev WITH PASSWORD 'dev123';
GRANT ALL PRIVILEGES ON DATABASE ibkr2_dev TO dev;

GRANT ALL ON ALL TABLES IN SCHEMA public TO dev;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dev;
GRANT ALL ON SCHEMA public TO dev;
GRANT CREATE ON SCHEMA public TO dev;
ALTER SCHEMA public OWNER TO dev;


# Setup

# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create PostgreSQL database
sudo -u postgres psql
CREATE DATABASE ohlcv_db;
CREATE USER your_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ohlcv_db TO your_user;
\q

# 4. Run migrations
python manage.py makemigrations
python manage.py migrate

# 5. Create superuser
python manage.py createsuperuser

# 6. Load data
# Load all data
python manage.py load_ohlcv_data

# Or load specific frequency/year
python manage.py load_ohlcv_data --frequency 1D --year 2026
python manage.py load_ohlcv_data --frequency 1D --year 2026 --data-dir ../ibkr/data/

# 7. Install and start Redis (for caching)
# Ubuntu/Debian:
sudo apt-get install redis-server
sudo systemctl start redis-server

# 8. Run development server
python manage.py runserver

# Tree
sudo apt update
sudo apt install tree 

tree
tree -d -I venv -I staticfiles


