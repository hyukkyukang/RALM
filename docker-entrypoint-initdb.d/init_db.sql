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
GRANT SELECT, INSERT, UPDATE ON metadata TO scraper;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO scraper;