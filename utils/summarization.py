"""
Text summarization utilities using flexible LLM providers.
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def chunk_text(text: str, max_length: Optional[int] = None) -> List[str]:
    """
    Split text into chunks by word count.

    Args:
        text: Input text to chunk
        max_length: Maximum words per chunk (from config if None)

    Returns:
        List of text chunks
    """
    from config import get_config

    if max_length is None:
        max_length = get_config().llm.chunk_size

    words = text.split()
    chunks = [
        ' '.join(words[i:i + max_length])
        for i in range(0, len(words), max_length)
    ]

    logger.info(f"Split text into {len(chunks)} chunks (max {max_length} words each)")
    return chunks


def summarize_text_chunk(
    text: str,
    video_info: Dict[str, Any],
    llm_provider
) -> str:
    """
    Summarize a text chunk with video context.

    Args:
        text: Text chunk to summarize
        video_info: Video metadata (title, author, description)
        llm_provider: LLM provider instance

    Returns:
        Summary text
    """
    # Build comprehensive prompt
    prompt = (
        f"Summarize the following video transcript chunk in a coherent and detailed manner. "
        f"Highlight key points and maintain the flow of the narrative.\n\n"
    )

    # Add video context if available
    if video_info.get('title'):
        prompt += f"Video Title: {video_info['title']}\n"
    if video_info.get('author'):
        prompt += f"Author: {video_info['author']}\n"
    if video_info.get('description'):
        prompt += f"Description: {video_info['description']}\n"

    prompt += f"\nTranscript:\n{text}\n\nProvide a clear, concise summary:"

    try:
        response = llm_provider.generate(
            prompt=prompt,
            max_tokens=150
        )

        # Extract summary from response
        # Handle different response formats
        if "summary:" in response.lower():
            summary = response.lower().split("summary:")[-1].strip()
        else:
            # Remove the prompt if it was echoed back
            summary = response.replace(prompt, "").strip()

        logger.debug(f"Generated summary: {len(summary)} characters")
        return summary

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise RuntimeError(f"Failed to summarize text: {e}")


def query_llm(
    question: str,
    context: str,
    llm_provider,
    max_tokens: Optional[int] = None
) -> str:
    """
    Query LLM with a question and context.

    Args:
        question: Question to ask
        context: Context information
        llm_provider: LLM provider instance
        max_tokens: Maximum tokens to generate

    Returns:
        Answer text
    """
    prompt = f"{question}\n\nContext:\n{context}\n\nAnswer:"

    try:
        response = llm_provider.generate(
            prompt=prompt,
            max_tokens=max_tokens or 200
        )

        # Extract answer from response
        if "answer:" in response.lower():
            answer = response.lower().split("answer:")[-1].strip()
        else:
            # Remove the prompt if it was echoed back
            answer = response.replace(prompt, "").strip()

        logger.debug(f"Generated answer: {len(answer)} characters")
        return answer

    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise RuntimeError(f"Failed to query LLM: {e}")


def generate_summary(
    transcript: str,
    video_info: Dict[str, Any],
    llm_provider
) -> Dict[str, str]:
    """
    Generate comprehensive summary from transcript.

    Args:
        transcript: Full video transcript
        video_info: Video metadata
        llm_provider: LLM provider instance

    Returns:
        Dictionary with 'full', 'short', and 'key_points' summaries
    """
    logger.info("Starting summarization pipeline")

    try:
        # Step 1: Chunk the transcript
        chunks = chunk_text(transcript)

        # Step 2: Summarize each chunk
        logger.info(f"Summarizing {len(chunks)} chunks")
        chunk_summaries = []

        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {i}/{len(chunks)}")
            summary = summarize_text_chunk(chunk, video_info, llm_provider)
            chunk_summaries.append(summary)

        # Step 3: Combine chunk summaries
        full_summary = " ".join(chunk_summaries).replace('\n', ' ').replace('  ', ' ')
        logger.info(f"Generated full summary: {len(full_summary)} characters")

        # Step 4: Generate short summary
        logger.info("Generating short summary")
        short_summary = query_llm(
            "What is this video about? Provide a detailed and coherent summary.",
            full_summary,
            llm_provider
        )

        # Step 5: Extract key points
        logger.info("Extracting key points")
        key_points = query_llm(
            "List 7-9 key points from this video.",
            full_summary,
            llm_provider
        )

        return {
            'full': full_summary,
            'short': short_summary,
            'key_points': key_points
        }

    except Exception as e:
        logger.error(f"Summarization pipeline failed: {e}")
        raise
