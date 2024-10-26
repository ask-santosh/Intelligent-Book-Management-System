from datetime import datetime
import enum
import re

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
    validates,
)
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Enum,
    create_engine
)

# SQLite async database setup
DATABASE_URL = "./library.db"
async_engine = create_async_engine("sqlite+aiosqlite:///" + DATABASE_URL, echo=True)
sync_engine = create_engine('sqlite:///'+DATABASE_URL, echo=True)

# Async SessionLocal with async capabilities
AsyncSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=async_engine, class_=AsyncSession
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


def as_dict(self):
    return {c.name: getattr(self, c.name) for c in self.__table__.columns}


Base = declarative_base()
Base.as_dict = as_dict


# Enum for genre list
class GenreEnum(enum.Enum):
    Biography = "Biography"
    Fiction = "Fiction"
    Fantasy = "Fantasy"
    Horror = "Horror"
    Literature = "Literature"
    Mystery = "Mystery"
    Religion = "Religion"
    Romance = "Romance"
    SciFi = "Sci-Fi"


# Book model
class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    author = Column(String(255), nullable=False, index=True)
    genre = Column(Enum(GenreEnum), nullable=False, index=True)
    year_published = Column(Integer, nullable=False)
    summary = Column(String(2500), nullable=True)
    source = Column(String(500), nullable=False)

    # Relationship to Reviews
    reviews = relationship("Review", back_populates="books")

    @validates("year_published")
    def validate_year(self, key, value):
        """Ensure the year is between 1800 and the current year"""
        current_year = datetime.now().year
        if not (1800 <= value <= current_year):
            raise ValueError(f"Year must be between 1800 and {current_year}.")
        return value


# User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    first_name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)

    # Relationship to Reviews
    reviews = relationship("Review", back_populates="users")

    @validates("email")
    def validate_email(self, key, value):
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(email_regex, value):
            raise ValueError("Invalid email address.")
        return value


# Reviews model
class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating = Column(Integer, nullable=False)
    review = Column(String(500))

    # Relationships
    books = relationship("Book", back_populates="reviews")
    users = relationship("User", back_populates="reviews")

    __table_args__ = (
        UniqueConstraint("book_id", "user_id", name="uq_book_user_review"),
    )

    @validates("rating")
    def validate_rating(self, key, value):
        """Ensure rating is an integer between 0 and 5"""
        if not isinstance(value, int) or not (0 <= value <= 5):
            raise ValueError("Rating must be an integer between 0 and 5.")
        return value


# Dependency to get the async session
async def get_async_db():
    global AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        yield session


# Create the tables asynchronously, only if they don't exist
async def init_db():
    global async_engine

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
