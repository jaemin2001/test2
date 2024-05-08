from uuid import UUID, uuid4
from typing import (Any, List)
from typing import Optional
from sqlalchemy import (Column, String, Integer, ForeignKey, Table)
from sqlalchemy.orm import (Mapped, mapped_column)
# from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.dialects.sqlite import UUID
# from sqlalchemy.dialects.postgresql import UUID

# Base = declarative_base()

class Base(DeclarativeBase):
    """Base database model."""
    # pk: Mapped[UUID] = mapped_column(
    #     primary_key=True,
    #     default=uuid4,
    # )
    pass

# base_association_table = Table(
#     "BASE_ASSOCIATION_TABLE",
#     Base.metadata,
#     # Column("CSCIDS2017_id", ForeignKey("CSCIDS2017.pk"), primary_key=True, default=str(uuid4()),),
#     # Column("test_id", UUID(as_uuid=True), ForeignKey("test.pk"))
# )


