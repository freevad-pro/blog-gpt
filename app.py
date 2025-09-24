import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# --- OpenAI (новый SDK) ---
from openai import OpenAI

app = FastAPI()

# Получаем API ключи из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

# Проверяем, что оба API ключа заданы
if not OPENAI_API_KEY or not CURRENTS_API_KEY:
    raise ValueError("Переменные окружения OPENAI_API_KEY и CURRENTS_API_KEY должны быть установлены")

# Инициализируем клиента OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

class Topic(BaseModel):
    topic: str

# Функция для получения последних новостей на заданную тему
def get_recent_news(topic: str) -> str:
    url = "https://api.currentsapi.services/v1/latest-news"
    params = {
        "language": "en",
        "keywords": topic,
        "apiKey": CURRENTS_API_KEY,
    }
    response = requests.get(url, params=params, timeout=20)
    if response.status_code != 200:
        # status_code должен быть int
        raise HTTPException(status_code=response.status_code,
                            detail=f"Ошибка при получении данных: {response.text}")

    news_data = response.json().get("news", [])
    if not news_data:
        return "Свежих новостей не найдено."
    return "\n".join([article.get("title", "").strip() for article in news_data[:5] if article.get("title")])

# Функция для генерации контента на основе темы и новостей
def generate_content(topic: str):
    recent_news = get_recent_news(topic)

    try:
        # Заголовок
        title_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Придумайте привлекательный и точный заголовок для статьи на тему '{topic}', "
                    f"с учётом актуальных новостей:\n{recent_news}. "
                    f"Заголовок должен быть интересным и ясно передавать суть темы."
                )
            }],
            max_tokens=60,
            temperature=0.5,
            stop=["\n"]
        )
        title = (title_resp.choices[0].message.content or "").strip()

        # Мета-описание
        meta_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Напишите мета-описание для статьи с заголовком: '{title}'. "
                    f"Оно должно быть полным, информативным и содержать основные ключевые слова."
                )
            }],
            max_tokens=120,
            temperature=0.5,
            stop=["."]  # при желании можно убрать, чтобы не обрезать по первой точке
        )
        meta_description = (meta_resp.choices[0].message.content or "").strip()

        # Полный контент
        post_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Напишите подробную статью на тему '{topic}', используя последние новости:\n{recent_news}.\n"
                    "Статья должна быть:\n"
                    "1. Информативной и логичной\n"
                    "2. Содержать не менее 1500 символов\n"
                    "3. Иметь четкую структуру с подзаголовками\n"
                    "4. Включать анализ текущих трендов\n"
                    "5. Иметь вступление, основную часть и заключение\n"
                    "6. Включать примеры из актуальных новостей\n"
                    "7. Каждый абзац должен быть не менее 3-4 предложений\n"
                    "8. Текст должен быть легким для восприятия и содержательным"
                )
            }],
            max_tokens=1500,
            temperature=0.5,
            presence_penalty=0.6,
            frequency_penalty=0.6
        )
        post_content = (post_resp.choices[0].message.content or "").strip()

        return {
            "title": title,
            "meta_description": meta_description,
            "post_content": post_content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации контента: {str(e)}")

@app.post("/generate-post")
async def generate_post_api(topic: Topic):
    return generate_content(topic.topic)

@app.get("/")
async def root():
    return {"message": "Service is running"}

@app.get("/heartbeat")
async def heartbeat_api():
    return {"status": "OK"}

if __name__ == "__main__":
    # Локальный запуск (на Koyeb запустит Procfile/Run command)
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
