import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from inngest.experimental import ai
from data_loader import load_and_chunk_pdf, embed_text
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    # Event that triggers this function
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)

async def rag_ingest_pdf(ctx: inngest.Context):
    return {"hello": "world"}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])

