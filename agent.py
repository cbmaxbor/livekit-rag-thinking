import logging
import pickle
import asyncio
import random
import wave

import chromadb
import numpy as np
from typing import Annotated
from pathlib import Path
from livekit import rtc

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, rag, silero, elevenlabs, turn_detector, noise_cancellation

load_dotenv(dotenv_path=".env.local")

logger = logging.getLogger("rag-assistant")

EMBEDDINGS_DIMENSION = 1536
INDEX_PATH = "vdb_data"
DATA_PATH = "my_data.pkl"

# Add chat context lock
_chat_ctx_lock = asyncio.Lock()

annoy_index = rag.annoy.AnnoyIndex.load(INDEX_PATH)
with open(DATA_PATH, "rb") as f:
    paragraphs_by_uuid = pickle.load(f)

# Cache for wav file data to avoid repeated disk reads
_wav_cache = {}
# Cache for audio track and source
_wav_audio_track = None
_wav_audio_source = None

db_client = chromadb.PersistentClient(path="./chroma")

collection = db_client.get_or_create_collection(name="realty_offers")

async def _enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext) -> None:
    """
    Locate the last user message, use it to query the RAG model for
    the most relevant paragraph, add that to context, and generate a response.
    """
    async with _chat_ctx_lock:
        user_msg = chat_ctx.messages[-1]

    # Let's sleep for 5 seconds to simulate a delay
    await asyncio.sleep(5)

    user_embedding = await openai.create_embeddings(
        input=[user_msg.content],
        model="text-embedding-3-small",
        dimensions=EMBEDDINGS_DIMENSION,
    )

    result = annoy_index.query(user_embedding[0].embedding, n=1)[0]
    paragraph = paragraphs_by_uuid[result.userdata]

    results = collection.query(
        query_texts=[user_msg.content],
        n_results=5
    )

    relevant_documents = "\n".join(results["documents"][0])

    if relevant_documents:
        logger.info(f"enriching with RAG: {relevant_documents}")
        rag_msg = llm.ChatMessage.create(
            text=f"""Context:\n```xml
       {relevant_documents}
       ```""",
            role="assistant",
        )
        
        async with _chat_ctx_lock:
            # Replace last message with RAG, then append user message at the end
            chat_ctx.messages[-1] = rag_msg
            chat_ctx.messages.append(user_msg)

            # Generate a response using the enriched context
            llm_stream = agent._llm.chat(chat_ctx=chat_ctx)
            await agent.say(llm_stream)


async def entrypoint(ctx: JobContext) -> None:
    """
    Main entrypoint for the agent. Sets up function context, defines
    RAG enrichment command, creates the agent's initial conversation context,
    and starts the agent.
    """
    fnc_ctx = llm.FunctionContext()

    agent = VoicePipelineAgent(
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=(
                "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
                "Keep responses short and concise. Avoid unpronounceable punctuation. "
                "Use any provided context to answer the user's question if needed."
                "Never start a sentence with phrases like 'Sure' or 'I can do that' or 'I can help with that'. Instead, just start with the answer."
                # "Option 1: Include this in the system prompt to make the agent say that it's looking up the answer w/ every function call. This doesn't always work, but is the simplest solution."
                # "If you need to perform a function call, always tell the user that you are looking up the answer."
            ),
        ),
        vad=silero.VAD.load(),
        stt=openai.stt.STT(detect_language=True, model='gpt-4o-mini-transcribe'),
        llm=openai.LLM(model="gpt-4o-mini-2024-07-18"),
        tts=elevenlabs.tts.TTS(
            model="eleven_turbo_v2_5",
            voice=elevenlabs.tts.Voice(
                id="EXAVITQu4vr4xnSDxMaL",
                name="Bella",
                category="premade",
                settings=elevenlabs.tts.VoiceSettings(
                    stability=0.71,
                    similarity_boost=0.5,
                    style=0.0,
                    use_speaker_boost=True
                ),
            ),
            language="en",
            streaming_latency=3,
            enable_ssml_parsing=False,
            chunk_length_schedule=[80, 120, 200, 260],
        ),
        turn_detector=turn_detector.EOUModel(),
        noise_cancellation=noise_cancellation.BVC(),
        fnc_ctx=fnc_ctx,
    )

    @fnc_ctx.ai_callable()
    async def enrich_with_rag(
        code: Annotated[
            int, llm.TypeInfo(description="Enrich with RAG for questions about LiveKit.")
        ]
    ):
        """
        Called when you need to enrich with RAG for questions about LiveKit.
        """
        logger.info("Enriching with RAG for questions about LiveKit")

        ############################################################
        # Options for thinking messages
        # Option 1 is included in the system prompt
        ############################################################

        # Option 2: Use a message from a specific list to indicate that we're looking up the answer
        thinking_messages = [
            "Let me look that up...",
            "One moment while I check...",
            "I'll find that information for you...",
            "Just a second while I search...",
            "Looking into that now..."
        ]
        await agent.say(random.choice(thinking_messages))

        # Option 3: Make a call to the LLM to generate a custom message for this specific function call
        # async with _chat_ctx_lock:
        #     thinking_ctx = llm.ChatContext().append(
        #         role="system",
        #         text="Generate a very short message to indicate that we're looking up the answer in the docs"
        #     )
        #     thinking_stream = agent._llm.chat(chat_ctx=thinking_ctx)
        #     # Wait for thinking message to complete before proceeding
        #     await agent.say(thinking_stream, add_to_chat_ctx=False)

        # Option 4: Play an audio file through the room's audio track
        # await play_wav_once("let_me_check_that.wav", ctx.room)

        ############################################################
        ############################################################

        await _enrich_with_rag(agent, agent.chat_ctx)

    # Connect and start the agent
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent.start(ctx.room)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))