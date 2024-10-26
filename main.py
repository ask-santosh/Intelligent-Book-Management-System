import aiofiles
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi import Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel, EmailStr, Field, conint, constr
import requests
import socket
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import subprocess
import sys
import threading
from typing import List, Literal, Optional
import uuid


from ai_helper import generate_summary

# Import async session and models
from models import Book, Review, User, GenreEnum, init_db, get_async_db

# Directory to store uploaded files
UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


# Async context manager for lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database when the app starts
    await init_db()
    yield
    # Here you could add logic for shutdown if necessary


# Initialize FastAPI
app = FastAPI(lifespan=lifespan)


### Pydantic models (request and response schema)
# Pydantic Model for BookCreate
class BookCreate(BaseModel):
    title: constr(strip_whitespace=True, min_length=1, max_length=255)
    author: constr(strip_whitespace=True, min_length=1, max_length=255)
    genre: Literal[[genre.value for genre in GenreEnum]]
    year_published: conint(ge=1800, le=datetime.now().year)
    book_file: UploadFile

    class Config:
        schema_extra = {
            "example": {
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "genre": "Fiction",
                "year_published": 1925,
                "book_file": "upload_file_here.pdf",
            }
        }


class BookShow(BaseModel):
    id: int
    title: str
    author: str
    genre: str
    year_published: int
    summary: str | None
    source: str


class BookUpdate(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    genre: Optional[str] = None
    year_published: Optional[int] = Field(None, ge=1800, le=datetime.now().year)


class UserCreate(BaseModel):
    first_name: constr(strip_whitespace=True, min_length=1, max_length=255)
    email: EmailStr = Field(..., description="User email, must be unique")

    class Config:
        schema_extra = {
            "example": {
                "first_name": "James",
                "email": "james@bond.com",
            }
        }

class UserShow(BaseModel):
    id: int
    first_name: str
    email: str


class ReviewCreate(BaseModel):
    user_id: int
    review_text: constr(strip_whitespace=True, min_length=10, max_length=500)
    rating: conint(ge=0, le=5)

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "review_text": "I liked the story.",
                "rating": 4,
            }
        }


# Add a new book
@app.post("/books", response_model=dict)
async def create_book(
    title: str = Form(..., strip_whitespace=True, min_length=1, max_length=255),
    author: str = Form(
        ..., strip_whitespace=True, min_length=1, max_length=255
    ),
    genre: str = Form(...),
    year_published: int = Form(..., ge=1800, le=datetime.now().year),
    book_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_db),
):
    genres = [genre.value for genre in GenreEnum]
    if genre not in genres:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid genre, Input should be {genres}",
        )

    # Validate file format
    if book_file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid file format, only PDF files are accepted.",
        )

    file_ext = book_file.filename.split(".")[-1]
    # Generate file path
    file_location = os.path.join(UPLOAD_DIRECTORY, f"{uuid.uuid4()}.{file_ext}")

    # Save the file asynchronously
    async with aiofiles.open(file_location, "wb") as f:
        await f.write(await book_file.read())

    # Create a new Book record in the database
    new_book = Book(
        title=title,
        author=author,
        genre=genre,
        year_published=year_published,
        source=file_location,
    )
    db.add(new_book)
    await db.commit()
    await db.refresh(new_book)  # Refresh to get the new book's ID

    thread = threading.Thread(
        target=generate_summary, args=(new_book.id, file_location, title)
    )
    thread.start()

    # Return a success message
    return {
        "message": f"Book '{title}' by {author} created successfully!",
        "book_details": new_book.as_dict(),
    }


# Get all books details
@app.get("/books", response_model=List[BookShow])
async def read_books(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1),
    db: AsyncSession = Depends(get_async_db),
):
    # Paginated query
    result = await db.execute(select(Book).offset(skip).limit(limit))
    books = result.scalars().all()

    return books


# Get a book details by ID
@app.get("/books/{book_id}", response_model=BookShow)
async def read_book(book_id: int, db: AsyncSession = Depends(get_async_db)):
    # Fetch the book by ID
    result = await db.execute(select(Book).filter(Book.id == book_id))
    book = result.scalar_one_or_none()

    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    return book


# Update a book information by ID
@app.put("/books/{book_id}", response_model=dict)
async def update_book(
    book_id: int,
    book_update: BookUpdate,
    db: AsyncSession = Depends(get_async_db),
):
    # Fetch the book by ID
    result = await db.execute(select(Book).filter(Book.id == book_id))
    fetched_book = result.scalar_one_or_none()

    if fetched_book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    # Update book details only for the fields that are provided
    if book_update.title is not None:
        fetched_book.title = book_update.title
    if book_update.author is not None:
        fetched_book.author = book_update.author
    if book_update.genre is not None:
        fetched_book.genre = book_update.genre
    if book_update.year_published is not None:
        fetched_book.year_published = book_update.year_published

    await db.commit()
    await db.refresh(fetched_book)

    return {
        "message": f"Book detail(s) updated successfully!",
        "book_details": fetched_book.as_dict(),
    }


# Delete a book by ID
@app.delete("/books/{book_id}", response_model=BookCreate)
async def delete_book(book_id: int, db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(Book).filter(Book.id == book_id))
    db_book = result.scalar_one_or_none()

    if db_book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    await db.delete(db_book)
    await db.commit()
    return db_book


# Create a user
@app.post("/users", response_model=dict)
async def create_user(
    user: UserCreate, db: AsyncSession = Depends(get_async_db)
):
    db_user = User(**dict(user))
    db.add(db_user)

    await db.commit()
    await db.refresh(db_user)

    # Return a success message
    return {
        "message": f"New user added!",
        "user_details": db_user.as_dict(),
    }


@app.get("/users", response_model=List[UserShow])
async def show_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1),
    db: AsyncSession = Depends(get_async_db),
):
    results = await db.execute(select(User).offset(skip).limit(limit))
    users=results.scalars().all()
    return users


@app.get("/users/{user_id}", response_model=UserShow)
async def show_user(user_id: int, db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user

        
    

# Create a review
@app.post("/books/{book_id}/reviews", response_model=dict)
async def create_review(
    book_id: int, review: ReviewCreate, db: AsyncSession = Depends(get_async_db)
):
    db_review = Review(**dict(review), book_id=book_id)

    db.add(db_review)
    await db.commit()
    await db.refresh(db_review)

    # Return a success message
    return {
        "message": f"New review added!",
        "review_details": db_review.as_dict(),
    }


# Get reviews for a specific book
@app.get("/books/{book_id}/reviews", response_model=dict)
async def read_reviews(book_id: int, db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(Review).filter(Review.book_id == book_id))
    reviews = result.scalars().all()

    print(reviews)

    return {}


@app.get("/uploads/{filename}", response_model=dict)
async def serve_static_uploaded_files(filename: str):
    file_path = os.path.join(UPLOAD_DIRECTORY, filename)
    return FileResponse(
        path=file_path, filename=file_path, media_type="application/pdf"
    )


# Function to run uvicorn asynchronously
async def run_uvicorn(host, port):
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()


# Function to check if the port is open
def check_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except socket.error:
            return False


# Function to run uvicorn and lt simultaneously
async def run_uvicorn_and_lt():
    host = "127.0.0.1"
    port = 8000
    custom_domain = "santosh-book"

    # Start uvicorn server in a background task
    server_task = asyncio.create_task(run_uvicorn(host, port))

    # Wait until the port is open
    retry_count = 0
    while not check_port_open(host, port):
        if retry_count > 10:
            print("Server didn't start within the expected time.")
            break
        print(
            f"Waiting for the server to start on port {port}... "
            f"(Attempt {retry_count + 1})"
        )
        await asyncio.sleep(1)  # Wait 1 second before retrying
        retry_count += 1

    print("Server is running. Starting lt...")

    # Run the lt command with the specified port and custom domain
    lt_command = f"lt -p {port} -s {custom_domain}"
    subprocess.Popen(lt_command, shell=True)

    ip_address = requests.get("https://ipv4.icanhazip.com/").text
    print("LocalTunnel is running!")
    print(f"Visit https://{custom_domain}.loca.lt to access api;")
    print(f"Use password {ip_address}\n")

    # Await the uvicorn server task to keep the program running
    await server_task


# Main block to run the application
if __name__ == "__main__":
    import uvicorn

    asyncio.run(init_db())  # Initialize the database

    try:
        asyncio.run(run_uvicorn_and_lt())
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
