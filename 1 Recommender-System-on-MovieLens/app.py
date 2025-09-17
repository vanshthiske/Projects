from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from models.recommender import ContentBasedRecommender
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI(
    title="Content-Based Movie Recommender",
    description="A beautiful movie recommendation system using TF-IDF and NearestNeighbors",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize recommender
recommender = ContentBasedRecommender()

@app.on_event("startup")
async def startup_event():
    """Load and prepare data on startup"""
    print("🎬 Loading movie data...")
    try:
        recommender.load_data("data/movies.csv")
        recommender.build_model()
        print("✅ Recommender system ready!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with search interface"""
    movie_titles = recommender.get_all_movie_titles()[:50]  # Show first 50 for dropdown
    return templates.TemplateResponse("index.html", {
        "request": request,
        "movie_titles": movie_titles
    })

@app.post("/recommend", response_class=HTMLResponse)
async def get_recommendations(request: Request, movie_title: str = Form(...)):
    """Get movie recommendations"""
    try:
        recommendations = recommender.get_recommendations(movie_title, num_recommendations=8)
        
        if not recommendations:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": f"Movie '{movie_title}' not found in database.",
                "movie_titles": recommender.get_all_movie_titles()[:50]
            })
        
        return templates.TemplateResponse("recommendations.html", {
            "request": request,
            "movie_title": movie_title,
            "recommendations": recommendations
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"An error occurred: {str(e)}",
            "movie_titles": recommender.get_all_movie_titles()[:50]
        })

@app.get("/api/search")
async def search_movies(query: str):
    """API endpoint for movie search autocomplete"""
    matches = recommender.search_movies(query, limit=10)
    return {"movies": matches}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "movies_loaded": len(recommender.movies_df) if recommender.movies_df is not None else 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
