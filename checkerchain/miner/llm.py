# OpenAI API Key (ensure this is set in env variables or a secure place)
from pydantic import BaseModel, Field
from checkerchain.types.checker_chain import UnreviewedProduct
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from checkerchain.utils.config import OPENAI_API_KEY


class ScoreBreakdown(BaseModel):
    """Detailed breakdown of product review scores."""

    project: int = Field(description="Innovation/Technology score")
    userbase: int = Field(description="Userbase/Adoption score")
    utility: int = Field(description="Utility Value score")
    security: int = Field(description="Security score")
    team: int = Field(description="Team evaluation score")
    tokenomics: int = Field(description="Price/Revenue/Tokenomics score")
    marketing: int = Field(description="Marketing & Social Presence score")
    roadmap: int = Field(description="Roadmap score")
    clarity: int = Field(description="Clarity & Confidence score")
    partnerships: int = Field(
        description="Partnerships (Collabs, VCs, Exchanges) score"
    )


class ReviewScoreSchema(BaseModel):
    """Structured output schema for product reviews."""

    product: str = Field(description="Product name")
    overall_score: int = Field(description="Overall review score out of 100")
    breakdown: ScoreBreakdown = Field(
        description="Breakdown of scores by evaluation criteria"
    )


async def create_llm():
    """
    Create an instance of the LLM with structured output.
    """
    try:
        model = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o",
            max_tokens=1000,
            temperature=0.5,  # Reduced temperature for more consistent evaluations
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n\n"],
        )
        return model.with_structured_output(ReviewScoreSchema)
    except Exception as e:
        raise Exception(f"Failed to create LLM: {str(e)}")


async def generate_review_score(product: UnreviewedProduct):
    """
    Generate review scores for a product using OpenAI's GPT.
    """
    prompt = f"""
    You are a highly experienced cryptocurrency and blockchain industry analyst with expertise in evaluating crypto products and projects. You have been tasked with conducting a comprehensive analysis of the following cryptocurrency/blockchain product and providing a detailed, objective trust score.

    **Product Details:**
    - Name: {product.name}
    - Description: {product.description}
    - Category: {product.category.name if hasattr(product.category, 'name') else product.category}
    - URL: {product.url}
    - Location: {product.location}
    - Network: {product.network}
    - Team: {len(product.teams)} members
    - Twitter Profile: {product.twitterProfile if product.twitterProfile else 'Not provided'}
    - Current Review Cycle: {product.currentReviewCycle}
    - Special Review Request: {product.specialReviewRequest if product.specialReviewRequest else 'None'}

    **Evaluation Framework - Score each criterion from 0-10:**

    1. Project (Innovation/Technology) - Evaluate:
       - Technical innovation and uniqueness in the blockchain space
       - Quality of technical implementation and architecture
       - Blockchain integration and utilization of the technology
       - Development activity and GitHub contributions (if available)

    2. Userbase/Adoption - Evaluate:
       - Current user base size and growth trajectory
       - Active users and engagement metrics
       - Adoption barriers and potential for mainstream use
       - Community growth and engagement levels

    3. Utility Value - Evaluate:
       - Real-world applications and use cases
       - Problem-solving capabilities
       - Value proposition and market need
       - Competitive advantage over alternatives

    4. Security - Evaluate:
       - Security audits and their results
       - History of vulnerabilities or exploits
       - Security practices and infrastructure
       - Risk management approach

    5. Team - Evaluate:
       - Team credentials and experience in blockchain
       - Leadership quality and track record
       - Transparency about team identity
       - Team size and composition relative to project scope

    6. Price/Revenue/Tokenomics - Evaluate:
       - Token utility and economics
       - Token distribution and supply dynamics
       - Revenue model sustainability
       - Value accrual mechanisms

    7. Marketing & Social Presence - Evaluate:
       - Brand visibility and recognition
       - Social media following and engagement
       - Marketing strategy effectiveness
       - Community building efforts

    8. Roadmap - Evaluate:
       - Clarity and detail of development roadmap
       - Feasibility of planned milestones
       - Track record of meeting deadlines
       - Vision and long-term planning

    9. Clarity & Confidence - Evaluate:
       - Quality and completeness of documentation
       - Transparency in operations and decisions
       - Communication clarity with community
       - Overall professionalism

    10. Partnerships - Evaluate:
        - Quality and relevance of partnerships
        - Collaboration with established entities
        - VC backing and investor quality
        - Exchange listings and liquidity

    **Scoring Guidelines:**
    - 0-2: Poor/Concerning (serious issues or red flags)
    - 3-4: Below Average (notable weaknesses)
    - 5-6: Average (meets basic industry standards)
    - 7-8: Above Average (strong implementation, exceeds standards)
    - 9-10: Exceptional (industry-leading, innovative excellence)

    Carefully analyze available information for each criterion. When information is limited, make reasonable inferences based on similar projects in the space, but don't assign high scores without evidence. Balance your assessment between optimism about potential and realistic evaluation of current status.

    All scores must be integers between 0-10.
    """

    system_message = """You are a leading cryptocurrency and blockchain analyst with 10+ years of experience evaluating blockchain projects. Your analyses are known for being comprehensive, balanced, and highly accurate in predicting project success and security. You excel at identifying both red flags and promising indicators, even with limited information. Your reputation depends on delivering trustworthy, data-driven evaluations that closely match consensus opinion from other expert reviewers."""

    try:
        llm = await create_llm()
        result = await llm.ainvoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt),
            ]
        )
        return result
    except Exception as e:
        raise Exception(f"Failed to generate review score: {str(e)}")
