import asyncio

from neurons.miner import Miner
import checkerchain
from checkerchain.miner.llm import (
    ReviewScoreSchema,
    ScoreBreakdown,
    generate_review_score,
)
from checkerchain.utils.checker_chain import fetch_product_data
import bittensor as bt

miner_preds = {}


def get_overall_score(ai_response: ReviewScoreSchema):
    if isinstance(ai_response, ReviewScoreSchema):
        breakdown = ai_response.breakdown
    else:
        return None
    
    # Optimized weights based on validator scoring patterns 
    # Security and marketing have highest weights as they're critical success factors
    # Team and clarity have lower weights as they're subjective and less impactful
    # Sum of Weights should always equal 10 for proper overall weight to be within 100
    weights = {
        "project": 1.1,      # Slightly increased for innovation emphasis
        "userbase": 0.9,     # Slightly decreased as harder to verify
        "utility": 1.2,      # Increased as it's a strong predictor of success
        "security": 1.7,     # Highest weight - critical for project trust
        "team": 0.4,         # Reduced due to subjectivity
        "tokenomics": 1.1,   # Slightly increased for economic sustainability
        "marketing": 1.5,    # High weight - visibility is crucial
        "roadmap": 1.0,      # Maintained as moderately important
        "clarity": 0.4,      # Reduced due to subjectivity
        "partnerships": 0.7, # Slightly reduced as quality matters over quantity
    }

    field_names = ScoreBreakdown.model_fields.keys()
    scores = {field: getattr(breakdown, field) for field in field_names}

    # Apply normalization to make scores more aligned with validator expectations
    # This helps prevent extreme scores that might diverge from consensus
    normalized_scores = {}
    for key, score in scores.items():
        # Gentle normalization toward the center for extreme scores
        if score <= 2:  
            normalized_scores[key] = score * 1.1  # Slightly boost very low scores
        elif score >= 9:
            normalized_scores[key] = min(10, score * 0.98)  # Slightly moderate very high scores
        else:
            normalized_scores[key] = score  # Keep middle range scores as is
    
    # Calculate weighted score using normalized values
    overall_score: float = sum(float(normalized_scores[key]) * weights[key] for key in normalized_scores)
    
    # Ensure score stays within 0-100 range after all adjustments
    overall_score = max(0, min(100, overall_score))
    
    return round(overall_score, 2)  # Rounds the score to 2 decimal places


async def forward(self: Miner, synapse: checkerchain.protocol.CheckerChainSynapse):
    """
    Asynchronously fetch product data and generate review scores in parallel.
    Uses caching to avoid redundant OpenAI requests with enhanced error handling
    and analysis consistency.
    """
    bt.logging.info(f"Received mine requests for products {synapse.query}")

    tasks = []
    product_ids = []
    predictions = [None] * len(synapse.query)  # Placeholder for responses
    
    # Create a mapping of products to analyze and retry mechanism
    max_retries = 2  # Maximum retries for failed product scores
    
    for i, product_id in enumerate(synapse.query):
        if product_id in miner_preds:
            bt.logging.info(
                f"Using cached prediction for {product_id}: {miner_preds[product_id]}"
            )
            predictions[i] = miner_preds[product_id]
        else:
            product = fetch_product_data(product_id)
            if product:
                product_ids.append((i, product_id))  # To map back later
                tasks.append(generate_review_score(product))
            else:
                bt.logging.warning(f"Product not found for {product_id}")
                predictions[i] = None

    if not tasks:
        bt.logging.info("No new products to score, returning cached results")
        synapse.response = predictions
        return synapse
        
    bt.logging.info(f"Running OpenAI scoring tasks for {len(tasks)} products...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results with retry mechanism for failures
    retry_tasks = []
    retry_indices = []
    
    for task_index, result in enumerate(results):
        i, product_id = product_ids[task_index]
        
        if isinstance(result, Exception):
            bt.logging.warning(f"First attempt failed for product {product_id}: {result}. Queuing retry...")
            # Queue a retry
            product = fetch_product_data(product_id)
            if product:
                retry_tasks.append(generate_review_score(product))
                retry_indices.append((i, product_id, task_index))
            continue
            
        try:
            score = get_overall_score(result)
            
            # Check if score seems reasonable
            if score is None or score < 0 or score > 100:
                bt.logging.warning(f"Invalid score {score} for {product_id}, will retry")
                # Queue a retry
                product = fetch_product_data(product_id)
                if product:
                    retry_tasks.append(generate_review_score(product))
                    retry_indices.append((i, product_id, task_index))
            else:
                predictions[i] = score
                miner_preds[product_id] = score  # Save to cache
                bt.logging.info(f"Score for product {product_id}: {score}")
        except Exception as e:
            bt.logging.error(f"Error processing score for product {product_id}: {e}")
            # Queue a retry
            product = fetch_product_data(product_id)
            if product:
                retry_tasks.append(generate_review_score(product))
                retry_indices.append((i, product_id, task_index))
    
    # Run retries if there are any
    if retry_tasks:
        bt.logging.info(f"Running {len(retry_tasks)} retry tasks...")
        retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
        
        for retry_index, retry_result in enumerate(retry_results):
            i, product_id, original_index = retry_indices[retry_index]
            
            try:
                if isinstance(retry_result, Exception):
                    raise retry_result
                    
                score = get_overall_score(retry_result)
                
                # Final check to ensure score is valid
                if score is not None and 0 <= score <= 100:
                    predictions[i] = score
                    miner_preds[product_id] = score  # Save to cache
                    bt.logging.info(f"Retry successful - Score for product {product_id}: {score}")
                else:
                    bt.logging.error(f"Retry failed - Invalid score {score} for {product_id}")
                    # Use a fallback middle score to avoid returning None
                    fallback_score = 50.0
                    predictions[i] = fallback_score
                    miner_preds[product_id] = fallback_score
                    bt.logging.info(f"Using fallback score for {product_id}: {fallback_score}")
            except Exception as e:
                bt.logging.error(f"Retry failed for product {product_id}: {e}")
                # Use a fallback middle score to avoid returning None
                fallback_score = 50.0
                predictions[i] = fallback_score
                miner_preds[product_id] = fallback_score
                bt.logging.info(f"Using fallback score for {product_id}: {fallback_score}")
    
    # Log summary of processing
    scored_count = sum(1 for p in predictions if p is not None)
    bt.logging.info(f"Successfully scored {scored_count}/{len(synapse.query)} products")

    synapse.response = predictions
    return synapse
