# import sys
# sys.path.append("D:/2024/Finance/NABI/NABI/core")

import os
from collections.abc import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from dotenv import load_dotenv
from core.config import Settings

configs = Settings()

# load_dotenv(dotenv_path="./core/.env",verbose=True)
# SQLALCHEMY_DATABASE_URL = os.environ.get('DATABASE_URL')
SQLALCHEMY_DATABASE_URL = configs.DATABASE_URL 
# engine = create_engine("sqlite+pysqlite:///:memory:", echo=True, future=True)

# fast_executemany=True 사용가능 확인
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
	engine = create_async_engine(SQLALCHEMY_DATABASE_URL) 
	factory = async_sessionmaker(autocommit=False,autoflush=False,bind=engine)
	async with factory() as session:
		try:
			yield session
			await session.commit()
		except SQLAlchemyError as error:
			await session.rollback()
			raise
