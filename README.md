📚 Course Recommendation Lab (Dockerized)

This repository contains a course recommendation system lab built to experiment with text embeddings and similarity-based recommendations.
The project is fully Dockerized, allowing it to run consistently on any machine without local dependency issues.

🚀 Tech Stack

Python

Sentence Transformers (Embeddings)

Docker

Docker Compose

📂 Project Structure
.
├── app/
│   ├── run_recommendation.py
│   ├── data/
│   ├── embeddings/
│   └── ...
├── docker/
│   └── Dockerfile
├── docker-compose.yml
└── README.md

🐳 Why Docker?

No need to install Python or libraries locally

Same runtime environment on every system

Easy to run after cloning the repository

Ideal for collaboration and deployment

⚙️ Prerequisites

Make sure the following are installed on your system:

Docker
https://docs.docker.com/get-docker/

Docker Compose (included with Docker Desktop)

Verify installation:

docker --version
docker compose version

📥 Clone the Repository
git clone <your-github-repo-url>
cd <your-repo-folder>

🧱 Docker Compose Configuration

This project uses the following docker-compose.yml file:

version: "3.9"

services:
  reco:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: course_reco_lab
    volumes:
      - ../app:/app
    stdin_open: true
    tty: true

▶️ Build and Run the Project
1️⃣ Build the Docker Image
docker compose build

2️⃣ Start the Container
docker compose up


The container will start in interactive mode.

🐚 Enter the Running Container

Open a new terminal and run:

docker exec -it course_reco_lab bash


You should now be inside the container:

root@<container-id>:/app#

▶️ Run the Recommendation Script

Inside the container, execute:

python run_recommendation.py


This command runs the recommendation logic and prints similarity-based course recommendations in the terminal.

🛑 Stop the Project

To stop and remove the container:

docker compose down

🧠 What This Project Demonstrates

Converting course text into vector embeddings

Similarity-based recommendation logic

Engineering-first approach to recommendation systems

Clean Docker workflow for reproducibility

📌 Future Enhancements

FAISS integration for fast vector search

API layer (FastAPI or Django)

Persistent vector storage

Integration with a full e-learning platform

📄 License

This project is intended for learning and experimentation purposes.