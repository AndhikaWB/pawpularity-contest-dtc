CREATE USER prefect WITH ENCRYPTED PASSWORD 'prefect123';
CREATE DATABASE prefect OWNER prefect;
\c prefect
-- Required by Prefect in order to work correctly
CREATE EXTENSION IF NOT EXISTS pg_trgm;