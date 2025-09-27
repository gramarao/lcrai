#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.rag_service_googleai import rag_service

async def test_system():
    """Test the fixed RAG system"""
    print("🧪 Testing Fixed RAG System...")
    
    try:
        # Test 1: Simple query
        print("\n1️⃣ Testing simple query...")
        result = await rag_service.search_and_generate(
            "What is artificial intelligence?",
            max_sources=3
        )
        print(f"✅ Query successful: {len(result.get('answer', ''))} chars")
        
        # Test 2: Streaming query
        print("\n2️⃣ Testing streaming query...")
        chunk_count = 0
        async for chunk in rag_service.stream_search_and_generate(
            "Explain machine learning",
            max_sources=3
        ):
            if chunk.get("type") == "content":
                chunk_count += 1
        print(f"✅ Streaming successful: {chunk_count} chunks")
        
        # Test 3: Document storage
        print("\n3️⃣ Testing document storage...")
        test_content = b"This is a test document about artificial intelligence and machine learning."
        doc_id = rag_service.store_document("test.txt", test_content)
        print(f"✅ Document stored: ID {doc_id}")
        
        print("\n🎉 All tests passed! System is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_system())
    sys.exit(0 if success else 1)