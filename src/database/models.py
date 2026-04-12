from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), nullable=False, index=True)
    series_id = Column(Integer, ForeignKey('product_series.id'), nullable=True)
    category = Column(String(100), index=True)
    subcategory = Column(String(100), index=True)
    primary_image_url = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    
    # Relationships
    series = relationship("ProductSeries", back_populates="products")
    specs = relationship("ProductSpec", back_populates="product", uselist=False)
    images = relationship("ProductImage", back_populates="product")

class ProductSeries(Base):
    __tablename__ = 'product_series'
    
    id = Column(Integer, primary_key=True, index=True)
    series_name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    
    products = relationship("Product", back_populates="series")

class ProductSpec(Base):
    __tablename__ = 'product_specs'
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete="CASCADE"), nullable=False, unique=True)
    
    # Core Specifications (Stored as JSON for arrays/ranges)
    wattage = Column(JSON, nullable=True) # e.g. [7, 12, 20]
    beam_angle = Column(JSON, nullable=True) # e.g. [24, 36]
    ip_rating = Column(String(50), index=True)
    driver = Column(JSON, nullable=True)
    led_type = Column(String(100), nullable=True)
    color_temperature = Column(JSON, nullable=True) # e.g. ["3000K", "4000K"]
    lumen_efficiency = Column(String(100), nullable=True)
    body_color = Column(JSON, nullable=True)
    material = Column(String(255), nullable=True)
    application = Column(String(255), index=True)
    mounting_type = Column(String(100), nullable=True)
    
    # Dimensions (Stored as JSON for multiple variants if applicable)
    diameter = Column(JSON, nullable=True)
    height = Column(JSON, nullable=True)
    cutout = Column(JSON, nullable=True)
    
    product = relationship("Product", back_populates="specs")

class ProductImage(Base):
    __tablename__ = 'product_images'
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete="CASCADE"), nullable=False)
    image_url = Column(Text, nullable=False)
    image_type = Column(String(50)) # e.g., 'product', 'diagram', 'application'
    page_number = Column(Integer, nullable=True)
    
    product = relationship("Product", back_populates="images")
