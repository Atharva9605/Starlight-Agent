import unittest
from src.retrieval.query_parser import extract_query_filters, build_sql_filters
from src.ingestion.embedder import Embedder

class TestPipeline(unittest.TestCase):
    def test_query_parser(self):
        query = "Need outdoor spotlight around 12W with narrow beam."
        
        filters = extract_query_filters(query)
        
        self.assertIn("category", filters)
        self.assertEqual(filters["category"].lower(), "spotlight")
        self.assertEqual(filters["environment"].lower(), "outdoor")
        
        # Wattage bounds checking
        self.assertIn("wattage", filters)
        self.assertIsNotNone(filters["wattage"])
        
        # Test SQL Builder
        sql_clauses = build_sql_filters(filters)
        self.assertTrue(any("category ILIKE" in c for c in sql_clauses))
        self.assertTrue(any("application ILIKE" in c for c in sql_clauses))

    def test_semantic_similarity(self):
        # We test that BGE correctly places identical concepts close to each other
        product_spec = "AQUA series spotlight outdoor lighting 12W IP65 narrow beam."
        query_exact = "I need an outdoor spotlight around 12W."
        query_irrelevant = "Looking for a cheap indoor LED strip 5m rgb."
        
        emb_product = Embedder.embed_text(product_spec)
        emb_query_exact = Embedder.embed_text(query_exact)
        emb_query_irrelevant = Embedder.embed_text(query_irrelevant)
        
        import numpy as np
        
        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
        sim_exact = cosine_similarity(emb_product, emb_query_exact)
        sim_irrelevant = cosine_similarity(emb_product, emb_query_irrelevant)
        
        print(f"Exact similarity: {sim_exact}")
        print(f"Irrelevant similarity: {sim_irrelevant}")
        
        self.assertGreater(sim_exact, sim_irrelevant)
        self.assertGreater(sim_exact, 0.6) # BGE usually scores highly related text >0.7

if __name__ == '__main__':
    unittest.main()
