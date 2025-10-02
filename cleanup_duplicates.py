#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from sqlalchemy import create_engine, text
from config.settings import settings
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
engine = create_engine(DATABASE_URL)

def cleanup_duplicate_embeddings():
    """Remove duplicate chunks while preserving document relationships - UUID compatible"""
    with engine.begin() as conn:  # Auto-commit transaction
        print("🔍 Analyzing duplicates...")
        
        # 1. First, check if content_hash column exists and populate if needed
        try:
            # Check if column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'document_chunks' AND column_name = 'content_hash'
            """))
            if not result.fetchone():
                print("⚠️  content_hash column not found. Adding it...")
                conn.execute(text("""
                    ALTER TABLE document_chunks 
                    ADD COLUMN content_hash VARCHAR(32)
                """))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_document_chunks_content_hash ON document_chunks(content_hash)"))
        except Exception as e:
            print(f"⚠️  Could not add content_hash column: {e}")
            print("Continuing without content-based deduplication...")
            return
        
        # Populate hashes if column is new/empty
        result = conn.execute(text("SELECT COUNT(*) FROM document_chunks WHERE content_hash IS NULL AND LENGTH(content) > 20"))
        null_count = result.scalar()
        if null_count > 0:
            print(f"🔄 Populating {null_count} content hashes... (this may take a while)")
            conn.execute(text("""
                UPDATE document_chunks 
                SET content_hash = MD5(content)
                WHERE content_hash IS NULL AND LENGTH(content) > 20
            """))
            print("✅ Hash population complete")
        
        # 2. Find duplicate chunks by content hash (UUID compatible)
        # Use MIN(id::text) instead of MIN(document_id) for UUIDs
        result = conn.execute(text("""
            SELECT 
                content_hash,
                COUNT(*) as duplicate_count,
                MIN(document_id::text) as primary_doc_id_text,
                STRING_AGG(DISTINCT document_id::text, ', ' ORDER BY document_id::text) as doc_ids
            FROM document_chunks 
            WHERE content_hash IS NOT NULL AND LENGTH(content) > 20 AND content_hash != ''
            GROUP BY content_hash 
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 20  -- Limit for safety
        """))
        
        duplicates = result.fetchall()
        print(f"\n📊 Found {len(duplicates)} content hashes with duplicates")
        
        if not duplicates:
            print("✅ No duplicate chunks found!")
            
            # Still run basic stats
            total_chunks = conn.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar()
            total_docs = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
            print(f"\n📈 Current stats:")
            print(f"  Documents: {total_docs}")
            print(f"  Chunks: {total_chunks}")
            print(f"  Chunks per doc: {total_chunks / total_docs:.1f}")
            return
        
        # Show preview
        print(f"\n🔍 Top 5 duplicate groups:")
        for i, (content_hash, dup_count, primary_doc_id_text, doc_ids) in enumerate(duplicates[:5], 1):
            print(f"  {i}. Hash: {content_hash[:8]}... | Chunks: {dup_count} | Docs: {doc_ids[:50]}...")
        
        # 3. Get confirmation for cleanup
        response = input(f"\n⚠️  This will delete {sum(row[1] for row in duplicates)} duplicate chunks. Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cleanup cancelled")
            return
        
        # 4. Cleanup phase
        total_chunks_before = conn.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar()
        total_docs_before = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        
        print(f"\n📊 Before cleanup:")
        print(f"  Documents: {total_docs_before:,}")
        print(f"  Chunks: {total_chunks_before:,}")
        
        deleted_chunks = 0
        processed_groups = 0
        
        # 5. Process each duplicate group
        for content_hash, dup_count, primary_doc_id_text, doc_ids in duplicates:
            try:
                # Convert UUID string back to UUID
                primary_doc_id = uuid.UUID(primary_doc_id_text) if primary_doc_id_text else None
                
                if not primary_doc_id:
                    print(f"⚠️  Skipping group {content_hash[:8]}... (invalid primary ID)")
                    continue
                
                # Delete chunks from non-primary documents
                doc_id_list = [uuid.UUID(doc_id.strip()) for doc_id in doc_ids.split(',') if doc_id.strip()]
                non_primary_docs = [doc_id for doc_id in doc_id_list if doc_id != primary_doc_id]
                
                if not non_primary_docs:
                    continue
                
                # Build IN clause for UUIDs
                doc_id_placeholders = ','.join([f":doc{i}" for i in range(len(non_primary_docs))])
                params = {f"doc{i}": doc_id for i, doc_id in enumerate(non_primary_docs)}
                params["hash"] = content_hash
                params["primary_doc_id"] = primary_doc_id
                
                # Delete duplicates
                delete_query = text(f"""
                    DELETE FROM document_chunks 
                    WHERE content_hash = :hash 
                    AND document_id != :primary_doc_id
                    AND document_id IN ({doc_id_placeholders})
                """)
                
                result = conn.execute(delete_query, params)
                deleted_this_group = result.rowcount
                deleted_chunks += deleted_this_group
                processed_groups += 1
                
                if deleted_this_group > 0:
                    print(f"  ✅ Group {processed_groups}: Hash {content_hash[:8]}... | Deleted {deleted_this_group} chunks")
                
                # Progress indicator
                if processed_groups % 5 == 0:
                    print(f"    Progress: {processed_groups}/{len(duplicates)} groups processed")
                
            except Exception as e:
                print(f"  ❌ Error processing group {content_hash[:8]}...: {e}")
                continue
        
        # 6. Clean up empty documents (documents with no chunks left)
        print(f"\n🧹 Cleaning up empty documents...")
        result = conn.execute(text("""
            DELETE FROM documents d
            WHERE NOT EXISTS (
                SELECT 1 FROM document_chunks dc WHERE dc.document_id = d.id
            )
            AND d.id NOT IN (
                SELECT MIN(d2.id) 
                FROM documents d2 
                GROUP BY MD5(d2.content)  -- Keep one version per content hash
            )
        """))
        deleted_docs = result.rowcount
        print(f"  ✅ Removed {deleted_docs} empty/orphaned documents")
        
        # 7. Final statistics
        total_chunks_after = conn.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar()
        total_docs_after = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        
        print(f"\n📊 After cleanup:")
        print(f"  Documents: {total_docs_after:,} ({total_docs_before - total_docs_after} removed)")
        print(f"  Chunks: {total_chunks_after:,} ({deleted_chunks} removed)")
        
        if total_chunks_before > 0:
            savings_percent = ((total_chunks_before - total_chunks_after) / total_chunks_before) * 100
            print(f"  Storage reduction: {savings_percent:.1f}%")
            print(f"  Chunks per doc: {total_chunks_after / total_docs_after:.1f}")
        
        # 8. Storage size comparison
        try:
            before_size = conn.execute(text("SELECT pg_size_pretty(pg_total_relation_size('document_chunks'))")).scalar()
            after_size = conn.execute(text("SELECT pg_size_pretty(pg_total_relation_size('document_chunks'))")).scalar()
            print(f"  Table size: {before_size} → {after_size}")
        except:
            print("  (Could not calculate table size)")
        
        # 9. Optimize database
        print("\n🔧 Optimizing database indexes...")
        try:
            conn.execute(text("REINDEX TABLE document_chunks"))
            conn.execute(text("ANALYZE document_chunks"))
            conn.execute(text("ANALYZE documents"))
            print("  ✅ Indexes optimized")
        except Exception as e:
            print(f"  ⚠️  Index optimization warning: {e}")
        
        print(f"\n🎉 Cleanup complete! {processed_groups} duplicate groups processed.")
        print("💡 Restart your services to use the cleaned data: ./run-script.sh restart")
        print("🔍 Test with: SELECT COUNT(*) FROM document_chunks;")

if __name__ == "__main__":
    cleanup_duplicate_embeddings()