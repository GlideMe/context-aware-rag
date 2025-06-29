# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""constants.py: File contains constants for graph rag"""

### GRAPH EXTRACTION
CHUNK_VECTOR_INDEX_NAME = "vector"
DROP_CHUNK_VECTOR_INDEX_QUERY = f"DROP INDEX {CHUNK_VECTOR_INDEX_NAME} IF EXISTS;"
CREATE_CHUNK_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (c:Chunk) ON c.embedding
"""
DROP_INDEX_QUERY = "DROP INDEX entities IF EXISTS;"
LABELS_QUERY = "CALL db.labels()"
FULL_TEXT_QUERY = (
    "CREATE FULLTEXT INDEX entities FOR (n{labels_str}) ON EACH [n.id, n.description];"
)
FILTER_LABELS = ["Chunk", "Document"]

HYBRID_SEARCH_INDEX_DROP_QUERY = "DROP INDEX keyword IF EXISTS;"
HYBRID_SEARCH_FULL_TEXT_QUERY = (
    "CREATE FULLTEXT INDEX keyword FOR (n:Chunk) ON EACH [n.text]"
)


### Vector graph search
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT = 40
VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH = 0.3
VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH = 0.9
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MINMAX_CASE = 20
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MAX_CASE = 40

VECTOR_GRAPH_SEARCH_QUERY_PREFIX = """
WITH node as chunk, score
// find the document of the chunk
MATCH (chunk)-[:PART_OF]->(d:Document)
WHERE CASE
    WHEN $uuid IS NOT NULL THEN d.uuid = $uuid
    ELSE true
END
// aggregate chunk-details
WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score
// fetch entities
CALL (chunks) { WITH chunks
UNWIND chunks as chunkScore
WITH chunkScore.chunk as chunk
"""

VECTOR_GRAPH_SEARCH_ENTITY_QUERY = """
    OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
    WITH e, count(*) AS numChunks
    ORDER BY numChunks DESC
    LIMIT {no_of_entites}

    WITH
    CASE
        WHEN e.embedding IS NULL OR ({embedding_match_min} <= vector.similarity.cosine($embedding, e.embedding) AND vector.similarity.cosine($embedding, e.embedding) <= {embedding_match_max}) THEN
            collect {{
                OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,1}}(:!Chunk&!Document)
                RETURN path LIMIT {entity_limit_minmax_case}
            }}
        WHEN e.embedding IS NOT NULL AND vector.similarity.cosine($embedding, e.embedding) >  {embedding_match_max} THEN
            collect {{
                OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,2}}(:!Chunk&!Document)
                RETURN path LIMIT {entity_limit_max_case}
            }}
        ELSE
            collect {{
                MATCH path=(e)
                RETURN path
            }}
    END AS paths, e
"""

VECTOR_GRAPH_SEARCH_QUERY_SUFFIX = """
    WITH apoc.coll.toSet(apoc.coll.flatten(collect(DISTINCT paths))) AS paths,
         collect(DISTINCT e) AS entities

    // De-duplicate nodes and relationships across chunks
    RETURN
        collect {
            UNWIND paths AS p
            UNWIND relationships(p) AS r
            RETURN DISTINCT r
        } AS rels,
        collect {
            UNWIND paths AS p
            UNWIND nodes(p) AS n
            RETURN DISTINCT n
        } AS nodes,
        entities
}

// Generate metadata and text components for chunks, nodes, and relationships
WITH d, avg_score,
     [c IN chunks | c.chunk.text] AS texts,
     [c IN chunks | {id: c.chunk.id, score: c.score, chunkIdx: c.chunk.chunkIdx, content: c.chunk.text}] AS chunkdetails,
     [n IN nodes | elementId(n)] AS entityIds,
     [r IN rels | elementId(r)] AS relIds,
     apoc.coll.sort([
         n IN nodes |
         coalesce(apoc.coll.removeAll(labels(n), ['__Entity__'])[0], "") + ":" +
         n.id +
         (CASE WHEN n.description IS NOT NULL THEN " (" + n.description + ")" ELSE "" END)
     ]) AS nodeTexts,
     apoc.coll.sort([
         r IN rels |
         coalesce(apoc.coll.removeAll(labels(startNode(r)), ['__Entity__'])[0], "") + ":" +
         startNode(r).id + " " + type(r) + " " +
         coalesce(apoc.coll.removeAll(labels(endNode(r)), ['__Entity__'])[0], "") + ":" + endNode(r).id
     ]) AS relTexts,
     entities

// Combine texts into response text
WITH d, avg_score, chunkdetails, entityIds, relIds,
     "Text Content:\n" + apoc.text.join(texts, "\n----\n") +
     "\n----\nEntities:\n" + apoc.text.join(nodeTexts, "\n") +
     "\n----\nRelationships:\n" + apoc.text.join(relTexts, "\n") AS text,
     entities

RETURN
    text,
    avg_score AS score,
    {
        length: size(text),
        source: d.uuid,
        chunkdetails: chunkdetails,
        entities : {
            entityids: entityIds,
            relationshipids: relIds
        }
    } AS metadata
"""

VECTOR_GRAPH_SEARCH_QUERY = (
    VECTOR_GRAPH_SEARCH_QUERY_PREFIX
    + VECTOR_GRAPH_SEARCH_ENTITY_QUERY.format(
        no_of_entites=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT,
        embedding_match_min=VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH,
        embedding_match_max=VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH,
        entity_limit_minmax_case=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MINMAX_CASE,
        entity_limit_max_case=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MAX_CASE,
    )
    + VECTOR_GRAPH_SEARCH_QUERY_SUFFIX
)

### CHAT TEMPLATES
CHAT_SYSTEM_TEMPLATE = """
You are analyzing warehouse surveillance data. The analysis is provided below in the Summary section.

CRITICAL RULES:
1. ONLY answer based on the surveillance analysis provided
2. ALL questions are about THIS specific warehouse surveillance data
3. NEVER give generic responses that are not in summary!
4. If information isn't in the analysis, say "That information is not available in the video analysis"

When asked "is there a forklift?" - search the analysis for forklift mentions and answer based on what you find.

### Video Summary:
<summary>
{context}
</summary>

Answer the user's question based ONLY on the analysis above.

"""

QUESTION_TRANSFORM_TEMPLATE = """
Given the below conversation, generate a search query to look up in order to get information relevant to the conversation.
Only provide information relevant to the context. Do not invent information.
Only respond with the query, nothing else.
"""

## CHAT QUERIES
VECTOR_SEARCH_TOP_K = 5

## CHAT SETUP
CHAT_SEARCH_KWARG_SCORE_THRESHOLD = 0.5
CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD = 0.10

QUERY_TO_DELETE_UUID_GRAPH = """
                                MATCH (d:Document {uuid:$uuid})
                                WITH d
                                // First handle chunks with entities
                                OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)-[:HAS_ENTITY]->(e)
                                WITH d, c, e
                                WHERE (c IS NOT NULL AND e IS NOT NULL) AND NOT EXISTS {
                                    MATCH (e)<-[:HAS_ENTITY]-(c2:Chunk)<-[:PART_OF]-(d2:Document)
                                    WHERE d2 <> d
                                }
                                // Explicitly filter out NULLs before collecting
                                WITH d,
                                     [x IN collect(c) WHERE x IS NOT NULL] as chunksWithEntities,
                                     [x IN collect(e) WHERE x IS NOT NULL] as orphanedEntities

                                // Then handle chunks without entities
                                OPTIONAL MATCH (d)<-[:PART_OF]-(cNoEntity:Chunk)
                                WHERE NOT EXISTS { (cNoEntity)-[:HAS_ENTITY]->() }
                                WITH d, chunksWithEntities, orphanedEntities,
                                     [x IN collect(cNoEntity) WHERE x IS NOT NULL] as chunksWithoutEntities

                                // Delete all collected nodes
                                UNWIND orphanedEntities as orphanedEntity
                                UNWIND chunksWithEntities + chunksWithoutEntities as chunkToDelete
                                DETACH DELETE orphanedEntity, chunkToDelete, d
                                """
