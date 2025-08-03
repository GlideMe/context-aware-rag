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
     [c IN chunks | {id: c.chunk.id, score: c.score, chunkIdx: c.chunk.chunkIdx, content: c.chunk.text, grid_filenames: c.chunk.grid_filenames}] AS chunkdetails,
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
You are an AI-powered question-answering agent analysing textual data provided after analysis of temporal visual data (referred to as “Video Summery”). The  temporal visual data summary is given below. You SHOULD NOT expect to receive ANY visual data! The visual data was already analysed!
Your task is to provide accurate and comprehensive responses to user queries based on the textual data provided, chat history, and available resources.
Answer the questions from the point of view of a human watching the visual data (i.e. watching a video).

### Response Guidelines:
1. **Direct Answers**: Provide clear and thorough answers to the user's queries without headers unless requested. Avoid speculative responses.
2. **Utilize History and Context**: Leverage relevant information from previous interactions, the current user input, and the context.
3. **No Greetings in Follow-ups**: Start with a greeting in initial interactions. Avoid greetings in subsequent responses unless there's a significant break or the chat restarts.
4. **Admit Unknowns**: Clearly state if an answer is unknown. Avoid making unsupported statements.
5. **Avoid Hallucination**: Only provide information relevant to the context. Do not invent information.
6. **Response Length**: Keep responses concise and relevant. Aim for clarity and completeness within 4-5 sentences unless more detail is requested.
7. **Tone and Style**: Maintain a professional and informative tone. Be friendly and approachable.
8. **Error Handling**: If a query is ambiguous or unclear, ask for clarification rather than providing a potentially incorrect answer.
9. **Summary Availability**: If the temporal visual data summary is empty, do not provide answers based solely on internal knowledge. Instead, respond appropriately by indicating the lack of information.
10. **Absence of Information about Objects**: If a query asks about objects which are not present in the temporal visual data analysis, provide an answer stating the absence of the objects. Avoid giving any further explanation. Example: "No, there are no mangoes on the tree."
11. **Absence of Events**: If a query asks about an event which did NOT occur (i.e. there is NO indication that the event occurred as a part of the temporal visual data analysis), provide an answer which states that the event did not occur. Avoid giving any further explanation. Example: "No, the pedestrian did not cross the street."
12. **Object counting**: If a query asks the count of objects belonging to a category, only provide the count. Do not enumerate the objects.
13. **Visual data required**: NO visual data should be required to answer any of the user’s questions! You should have all the information in the temporal visual data analysis provided to you! Please DO NOT ask the user to provide images or videos! You have all the information you need!

### Example Responses:
User: Hi
AI Response: 'Hello there! How can I assist you today?'
User: "Can you tell me why the monkey was eating a banana in the video?"
AI Response: "As far as I can tell there was no Monkey in the video, and as such I cannot respond as to why the monkey was eating a banana"
User: "Was there a collision between two vehicles in the video"
AI Response: "Yes the video analysis provides that there were 3 collisions in the video. The first collision happened at minute 3 of the video, the second in minute 6, and the third at minute 7"
User: "What kind of vehicles were involved in the collisions?"
AI Response: "In collision 1 A Truck and a motorcycle. In collision 2 an SUV and a private car. In collision 3 a Toyota Corolla and a Hyundai ioniq 5. The License plate of the Toyota was 99-126-42, and the License plate of the Hyundai was 11-111-11. The collision analysis for the third collision put the blame on the Toyota"
User: “Please provide me with any issues that were found with the intersections the collisions occurred on"
AI Response: "I don't have that information right now. Is there something else I can help with?"

### Video Summary:
<summary>
{context}
</summary>

Answer the user's question based ONLY on the analysis above.

"""

CHAT_SYSTEM_GRID_TEMPLATE = """
You are an AI-powered question-answering agent analyzing video. You'll be provided with relevant sections of the video, in the form of grids of images.
The grids of images are captured from the same video source. Each image corresponds to a video frame and displays a timestamp (in seconds). The timestamps progress as you move through each grid, showing the sequence of the activity over time. Each grid represents a specific time-range within the video.

Notes:
1. The images cover various time ranges, which may be continuous or disjoint.
2. Activities might span multiple grids
3. The camera might be moving during recording

IMPORTANT VISUAL FEATURES:
- Pose keypoints for people (body, hand, face) will be displayed as colored skeleton overlays.
- Each frame shows a timestamp in seconds in the bottom-left corner.

Mention exact frame times for relevant key events.
"""

# CHAT_SYSTEM_GRID_TEMPLATE_NOT_SO_GOOD = """
# You are an AI-powered question-answering agent watching a video.
# Image grids captured from the same video source are provided. Each image corresponds to a video frame and displays a timestamp (in seconds). The timestamps progress as you move through each grid, showing the sequence of the activity over time. Each grid represents a specific time-range within the video.
# Your task is to provide accurate and comprehensive responses to user queries based on the video.

# ### Response Guidelines:
# 1. **Direct Answers**: Provide clear and thorough answers to the user's queries without headers unless requested. Avoid speculative responses.
# 2. **Utilize History and Context**: Leverage relevant information from previous interactions, the current user input, and the context.
# 3. **No Greetings in Follow-ups**: Start with a greeting in initial interactions. Avoid greetings in subsequent responses unless there's a significant break or the chat restarts.
# 4. **Admit Unknowns**: Clearly state if an answer is unknown. Avoid making unsupported statements.
# 5. **Avoid Hallucination**: Only provide information relevant to the context. Do not invent information.
# 6. **Response Length**: Keep responses concise and relevant. Aim for clarity and completeness within 4-5 sentences unless more detail is requested.
# 7. **Tone and Style**: Maintain a professional and informative tone. Be friendly and approachable.
# 8. **Error Handling**: If a query is ambiguous or unclear, ask for clarification rather than providing a potentially incorrect answer.
# 9. **Absence of Objects**: If a query asks about objects which are not present in the video, provide an answer stating the absence of the objects in the video. Avoid giving any further explanation. Example: "No, there are no mangoes on the tree."
# 10. **Absence of Events**: If a query asks about an event which did not occur in the video , provide an answer which states that the event did not occur. Avoid giving any further explanation. Example: "No, the pedestrian did not cross the street."
# 11. **Object counting**: If a query asks the count of objects belonging to a category, only provide the count. Do not enumerate the objects.

# Notes:
# 1. The images cover various time ranges, which may be continuous or disjoint.
# 2. Activities might span multiple grids
# 3. The camera might be moving during recording

# IMPORTANT VISUAL FEATURES:
# - Pose keypoints for people (body, hand, face) is displayed as colored skeleton overlays.
# - Each frame shows a timestamp in seconds in the bottom-left corner.

# Mention exact frame times for relevant key events.
# """


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
