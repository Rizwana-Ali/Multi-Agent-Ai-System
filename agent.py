from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
import os


load_dotenv()
set_tracing_disabled(True)

provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash-exp",
    openai_client=provider,
)

#Web Ddeveloper
web_dev = Agent(
    name = "Web Developer Expert",
    instructions="Build responsive and performant websites using modern frameworks",
    model=model,
    handoff_description="handoff to web developer if the task is relted to web development."
 
)

# app developer
mobile_dev = Agent(
    name = " mobile APP Developer Expert",
    instructions="Develope cross-platform mobile apps for ios and Android", 
    model=model,
    handoff_description="handoff to mobile app developer if the task is relted to mobile app development."
 
)

# Marketing agent
marketing_agent = Agent(
        name = "Marketing Agent",
        instructions="Create and execute marketing strategies to promote product and services", 
        model=model,
        handoff_description="handoff to marketing agent if the task is related  to marketing."

)

async def myAgent(user_input):
    manager = Agent(
        name="Manager",
        instructions="You will chat with the user and delegate task to the specialized agents based on their requests",
        model=model,
        handoff=[web_dev, mobile_dev, marketing_agent]

    )

    response = await Runner.run(
        manager,
        input=user_input,
    )

    return response.final_output


