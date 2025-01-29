-- Create database
CREATE DATABASE scrape;
\c scrape;

-- Set time zone
SET TIME ZONE 'Asia/Seoul';

-- Create tables
CREATE TABLE IF NOT EXISTS metadata (
            fid int not null primary key,
            url varchar(2048) not null,
            domain varchar(255) not null,
            word_count int null,
            elapsed int null,
            success boolean not null
    );

-- Create user
DROP ROLE IF EXISTS scraper;
CREATE USER scraper WITH PASSWORD 'scraper_pw';

-- Grant privileges
GRANT CONNECT ON DATABASE scrape TO scraper;
GRANT USAGE ON SCHEMA public TO scraper;

-- Grant full privileges on metadata table
ALTER TABLE metadata OWNER TO scraper;
GRANT ALL PRIVILEGES ON TABLE metadata TO scraper;

-- Grant sequence privileges (for auto-incremented columns, if any)
GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO scraper;

-- Ensure future tables & sequences in schema public get privileges automatically
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT ALL PRIVILEGES ON TABLES TO scraper;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO scraper;