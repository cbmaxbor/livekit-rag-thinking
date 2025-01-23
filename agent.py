import logging
import pickle
import asyncio
import random
import wave
import numpy as np
from typing import Annotated
from pathlib import Path
from livekit import rtc

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, rag, silero

load_dotenv()

logger = logging.getLogger("rag-assistant")

EMBEDDINGS_DIMENSION = 1536
INDEX_PATH = "vdb_data"
DATA_PATH = "my_data.pkl"


annoy_index = rag.annoy.AnnoyIndex.load(INDEX_PATH)
with open(DATA_PATH, "rb") as f:
    paragraphs_by_uuid = pickle.load(f)

async def _enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext) -> None:
    """
    Locate the last user message, use it to query the RAG model for
    the most relevant paragraph, add that to context, and generate a response.
    """
    user_msg = chat_ctx.messages[-1]

    # Let's sleep for 3 seconds to simulate a delay
    await asyncio.sleep(3)

    user_embedding = await openai.create_embeddings(
        input=[user_msg.content],
        model="text-embedding-3-small",
        dimensions=EMBEDDINGS_DIMENSION,
    )

    result = annoy_index.query(user_embedding[0].embedding, n=1)[0]
    paragraph = paragraphs_by_uuid[result.userdata]

    if paragraph:
        logger.info(f"enriching with RAG: {paragraph}")
        rag_msg = llm.ChatMessage.create(
            text="Context:\n" + paragraph,
            role="assistant",
        )
        # Replace last message with RAG, then append user message at the end
        chat_ctx.messages[-1] = rag_msg
        chat_ctx.messages.append(user_msg)

        # Generate a response using the enriched context
        llm_stream = agent._llm.chat(chat_ctx=chat_ctx)
        await agent.say(llm_stream)

async def play_wav_once(wav_path: str | Path, room: rtc.Room, volume: float = 0.3):
    """
    Simple function to play a WAV file once through a LiveKit audio track
    This is only needed for the "Option 3" thinking messagein the entrypoint function.
    """
    samples_per_channel = 9600
    wav_path = Path(wav_path)
    
    # Create audio source and track
    audio_source = rtc.AudioSource(48000, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("wav_audio", audio_source)
    
    # Publish the track
    await room.local_participant.publish_track(
        audio_track,
        rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE,
            stream="wav_audio"
        )
    )
    
    # Small delay to ensure track is established
    await asyncio.sleep(0.5)
    
    with wave.open(str(wav_path), 'rb') as wav_file:
        audio_data = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        if wav_file.getnchannels() == 2:
            audio_array = audio_array.reshape(-1, 2).mean(axis=1)
        
        for i in range(0, len(audio_array), samples_per_channel):
            chunk = audio_array[i:i + samples_per_channel]
            
            if len(chunk) < samples_per_channel:
                chunk = np.pad(chunk, (0, samples_per_channel - len(chunk)))
            
            chunk = np.tanh(chunk / 32768.0) * 32768.0
            chunk = np.round(chunk * volume).astype(np.int16)
            
            await audio_source.capture_frame(rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=48000,
                samples_per_channel=samples_per_channel,
                num_channels=1
            ))
            
            await asyncio.sleep((samples_per_channel / 48000) * 0.98)

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
            ),
        ),
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        fnc_ctx=fnc_ctx,
    )

    @fnc_ctx.ai_callable()
    async def enrich_with_rag(
        code: Annotated[
            int, llm.TypeInfo(description="Enrich with RAG for questions about LiveKit")
        ]
    ):
        """Called when you need to enrich with RAG for questions about LiveKit."""
        logger.info("Enriching with RAG for questions about LiveKit")

        ############################################################
        # Options for thinking messages
        ############################################################

        # Option 1: Use a message from a specific list to indicate that we're looking up the answer
        # thinking_messages = [
        #     "Let me look that up...",
        #     "One moment while I check...",
        #     "I'll find that information for you...",
        #     "Just a second while I search...",
        #     "Looking into that now..."
        # ]
        # await agent.say(random.choice(thinking_messages))

        # Option 2: Make a call to the LLM with a copied context
        # temp_ctx = agent.chat_ctx.copy().append(
        #     role="system",
        #     text="Generate a message to indicate that we're looking up the answer"
        # )
        # thinking_stream = agent._llm.chat(chat_ctx=temp_ctx)
        # await agent.say(thinking_stream, add_to_chat_ctx=False)

        # Option 3: Play an audio file through the room's audio track
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