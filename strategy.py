MISTAL_API_KEY = "ваш АРi"

from mistralai import Mistral
import base64
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod

class RequestStrategy(ABC):
    """
    Абстрактный класс, определяющий интерфейс для всех стратегий запросов.
    """

    @abstractmethod
    def execute(self, text: str, model: str, history: list = None, image_path: str = None) -> dict:
        """
        Абстрактный метод для выполнения запроса.
        Должен быть реализован в конкретных стратегиях.
        """
        pass

class TextRequestStrategy(RequestStrategy):
    """
    Конкретная реализация стратегии для отправки текстовых запросов.
    """
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = Mistral(api_key=self.api_key)

    def execute(self, text: str, model: str, history: list = None, image_path: str = None) -> dict:
        """
        Реализует отправку текстового запроса к API Mistral.
        """
        messages = []
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": text})

        response = self.client.chat.complete(
            model=model,
            messages=messages
        )

        # Формируем ответ в виде словаря для работы с историей чата
        result = {
            "role": "assistant",
            "content": response.choices[0].message.content
        }

        return result

class ImageRequestStrategy(RequestStrategy):
    """
    Конкретная реализация стратегии для отправки запросов с изображением.
    """
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = Mistral(api_key=self.api_key)

    def __encode_image(self, image_path: str) -> str:
        """Переводит изображение в формат base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: the file {image_path} was not found.")
            return ""
        except Exception as e:
            print(f"Error: {e}")
            return ""

    def execute(self, text: str, model: str, history: list = None, image_path: str = None) -> dict:
        """
        Реализует отправку мультимодального запроса, объединяющего текст и изображение.
        """
        if not image_path:
            return {"error": "Image path is required for ImageRequestStrategy."}

        # Получаем изображение в формате base64
        base64_image = self.__encode_image(image_path)
        if not base64_image:
            return {"error": "Failed to encode image."}

        # Формируем сообщение для чата
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]

        chat_response = self.client.chat.complete(
            model=model,
            messages=messages
        )

        # Формируем ответ в виде словаря для работы с историей чата
        result = {
            "role": "assistant",
            "content": chat_response.choices[0].message.content
        }

        return result

class MistralRequestContext:
    """
    Контекст для работы со стратегиями запросов к Mistral
    Реализуется паттерн Стратегия
    """
    def __init__(self, strategy: RequestStrategy) -> None:
        self.strategy = strategy

    def execute_strategy(self, text: str, model: str, history: list = None, image_path: str = None) -> dict:
        return self.strategy.execute(text, model, history, image_path)

class ChatFacade:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = {
            "text": ["mistral-large-latest"],
            "image": ["pixtral-12b-2409"]
        }
        self.text_strategy = TextRequestStrategy(api_key=self.api_key)
        self.image_strategy = ImageRequestStrategy(api_key=self.api_key)
        self.request_context = MistralRequestContext(self.text_strategy)
        self.model = self.select_model()
        self.history = []

    def change_strategy(self, strategy_type: str) -> None:
        """
        Метод для смены текущей стратегии запроса.
        """
        if strategy_type == "text":
            self.request_context = MistralRequestContext(self.text_strategy)
        elif strategy_type == "image":
            self.request_context = MistralRequestContext(self.image_strategy)
        else:
            raise ValueError("Неверный тип стратегии. Выберите 'text' или 'image'.")

    def select_model(self) -> str:
        """
        Позволяет выбрать модель из списка, соответствующую текущей стратегии.
        """
        model_type = 'text' if isinstance(self.request_context.strategy, TextRequestStrategy) else 'image'
        model = input(f"Выберите модель из списка {self.models[model_type]}: ")
        if model not in self.models[model_type]:
            raise ValueError('Неверная модель')
        return model

    def ask_question(self, text: str, model: str, image_path: str = None) -> dict:
        """
        Основной метод для отправки запроса.
        Делегирует выполнение запроса текущей стратегии.
        """
        # Получаем текущую историю
        current_history = [msg for _, msg in self.history]

        response = self.request_context.execute_strategy(text=text, model=model, history=current_history, image_path=image_path)

        # Обновляем историю
        self.history.append((text, {"role": "user", "content": text}))
        self.history.append((text, response))
        return response

    def get_history(self) -> list[tuple[str, dict]]:
        """
        Возвращает историю запросов и ответов.
        """
        return self.history

    def clear_history(self) -> None:
        """
        Очищает историю сообщений.
        """
        self.history.clear()

    def __call__(self):
        """
        Запуск фасада
        """
        print("Здравствуйте! Я готов помочь вам. Для выхода введите exit")
        while True:
            text = input("\nВведите текст запроса: ")
            if text.lower() == "exit":
                print('До свидания!')
                break

            strategy_type = input("Введите тип стратегии (text или image): ")
            self.change_strategy(strategy_type)

            model = self.select_model()
            image_path = None
            if strategy_type == "image":
                image_path = input("Введите путь к изображению: ")

            response = self.ask_question(text=text, model=model, image_path=image_path if image_path else None)

            # Выводим последний ответ
            print(response)

# Пример использования
chat_facade = ChatFacade(api_key=MISTAL_API_KEY)
chat_facade()
