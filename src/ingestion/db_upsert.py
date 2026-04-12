import sys
import os
import uuid
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import from other modules
from src.database.models import Base, Product, ProductSeries, ProductSpec, ProductImage
from src.database.qdrant_db import get_qdrant_client, COLLECTION_TEXT, COLLECTION_SPEC, COLLECTION_IMAGE
from qdrant_client.models import PointStruct

log = logging.getLogger("upsert")

class PipelineUpsert:
    def __init__(self, db_url="postgresql://starlight:starlight_password@localhost:5432/starlight_catalog"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.qdrant = get_qdrant_client()

    def upsert_product(self, product_data: dict, text_emb: list, spec_emb: list, images_data: list = None):
        """
        Takes raw dictionaries of parsed/embedded data and inserts them correctly
        into PostgreSQL (structured) and Qdrant (vectors).
        """
        session = self.SessionLocal()
        try:
            # 1. Handle Series
            series_name = product_data.get('series', '').strip()
            series_id = None
            if series_name:
                series = session.query(ProductSeries).filter_by(series_name=series_name).first()
                if not series:
                    series = ProductSeries(series_name=series_name)
                    session.add(series)
                    session.commit()
                    session.refresh(series)
                series_id = series.id

            # 2. Add Product
            prod_name = product_data.get('product_name', 'Unknown Product')
            product = session.query(Product).filter_by(product_name=prod_name).first()
            if not product:
                product = Product(
                    product_name=prod_name,
                    series_id=series_id,
                    category=product_data.get('category'),
                    subcategory=product_data.get('subcategory')
                )
                session.add(product)
                session.commit()
                session.refresh(product)
            else:
                # Update existing product
                product.series_id = series_id
                product.category = product_data.get('category')
                product.subcategory = product_data.get('subcategory')
                session.commit()

            # 3. Add Structured Specs
            spec = session.query(ProductSpec).filter_by(product_id=product.id).first()
            spec_data_map = {k: v for k, v in product_data.items() if k not in ['product_name', 'series', 'category', 'subcategory']}
            
            if not spec:
                spec = ProductSpec(product_id=product.id, **spec_data_map)
                session.add(spec)
            else:
                for key, value in spec_data_map.items():
                    setattr(spec, key, value)
            session.commit()

            # 4. Handle Images (SQL)
            if images_data:
                for img in images_data:
                    # check if exists
                    existing_img = session.query(ProductImage).filter_by(
                        product_id=product.id, image_url=img['url']
                    ).first()
                    if not existing_img:
                        new_img = ProductImage(
                            product_id=product.id,
                            image_url=img['url'],
                            image_type=img.get('type', 'product')
                        )
                        session.add(new_img)
                session.commit()

            # 5. Insert Vectors to Qdrant
            # We use a UUID based on the product ID to keep it consistent
            vec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"starlight_prod_{product.id}"))
            
            payload = {
                "product_id": product.id,
                "product_name": product.product_name,
                "category": product.category
            }
            
            # Upsert Text Embedding
            if text_emb:
                self.qdrant.upsert(
                    collection_name=COLLECTION_TEXT,
                    points=[PointStruct(id=vec_id, vector=text_emb, payload=payload)]
                )
                
            # Upsert Spec Embedding
            if spec_emb:
                self.qdrant.upsert(
                    collection_name=COLLECTION_SPEC,
                    points=[PointStruct(id=vec_id, vector=spec_emb, payload=payload)]
                )
                
            # Upsert Image Embeddings
            if images_data:
                image_points = []
                for idx, img in enumerate(images_data):
                    if 'embedding' in img and img['embedding']:
                        # Unique ID for each image
                        img_vec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"img_{product.id}_{idx}"))
                        img_payload = payload.copy()
                        img_payload['image_url'] = img['url']
                        image_points.append(
                            PointStruct(id=img_vec_id, vector=img['embedding'], payload=img_payload)
                        )
                if image_points:
                    self.qdrant.upsert(
                        collection_name=COLLECTION_IMAGE,
                        points=image_points
                    )
            
            log.info(f"Successfully upserted {prod_name} (ID: {product.id}) database and vector stores.")
            return True
            
        except Exception as e:
            session.rollback()
            log.error(f"Error upserting product: {e}")
            raise
        finally:
            session.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
