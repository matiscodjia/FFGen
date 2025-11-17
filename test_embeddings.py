from openai import AsyncOpenAI
import asyncio

async_client = None

def init_client():
    global async_client
    if async_client is None:
        async_client = AsyncOpenAI(
            api_key="none",
            base_url="http://localhost:8000/v1"
    )

async def encode(messages):
    response = await async_client.embeddings.create(
        model="default",
        input=messages
    )
    print([item.embedding for item in response.data])


async def main():
    messages = ["Test", "Encode me if you can"]
    init_client()
    try:
        await encode(messages)
    except Exception as e:
        print(f"Error : {e}")

if __name__=="__main__":
    asyncio.run(main())
