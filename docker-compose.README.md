# Docker Compose Setup for Mood Maestro

This document explains how to run the Mood Maestro application using Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- `.env` file with Azure OpenAI credentials (see `.env` file in project root)

## Quick Start

### 1. Production Setup
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### 2. Development Setup
```bash
# Start with development overrides (includes hot reload)
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Or simply (override file is loaded automatically)
docker-compose up -d

# Development mode includes hot reload and debugging ports
```

## Services

### Application (Port 8000)
- **URL**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Database
- **MongoDB Atlas**: Cloud-managed MongoDB (configured via MONGO_URI in .env)

## Environment Variables

The application requires these credentials in your `.env` file:

```env
# MongoDB Atlas
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/mood_maestro_db?retryWrites=true&w=majority
DB_NAME=mood_maestro_db

# Collections (optional - defaults provided)
MONGO_TRACKS_COLLECTION=tracks
MONGO_GENRES_COLLECTION=genres
MONGO_ARTISTS_COLLECTION=artists
MONGO_ALBUMS_COLLECTION=albums
MONGO_USERS_COLLECTION=users

# Azure OpenAI
AZURE_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview

# Features and data (optional - defaults provided)
EMBEDDING_FEATURES="duration_ms,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo"
DATA_FILE_PATH="dataset/final_full_dataset.csv"
```

## Data Setup

If you have a dataset file, place it in the `dataset/` directory:
```bash
mkdir -p dataset
# Copy your final_full_dataset.csv to dataset/
```

## Useful Commands

```bash
# Rebuild application container
docker-compose build app

# View application logs
docker-compose logs -f app

# MongoDB Atlas is managed externally - no local shell access needed

# Access application container
docker-compose exec app bash

# Clean up everything (including volumes)
docker-compose down -v
docker system prune -f
```

## Troubleshooting

### Application won't start
1. Check if `.env` file exists with MongoDB Atlas URI and Azure OpenAI credentials
2. Verify MongoDB Atlas connection string is correct
3. Check application logs: `docker-compose logs app`

### MongoDB Atlas connection issues
1. Ensure your MongoDB Atlas cluster is running and accessible
2. Verify the MONGO_URI in your `.env` file is correct
3. Check if your IP address is whitelisted in MongoDB Atlas
4. Confirm database user has proper permissions

### Performance optimization
- MongoDB Atlas provides managed performance optimization
- Adjust container resource limits in docker-compose.yml if needed
- Use Docker BuildKit for faster builds: `DOCKER_BUILDKIT=1 docker-compose build`
