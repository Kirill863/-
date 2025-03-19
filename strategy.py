MISTAL_API_KEY = "ваш АРi"

from mistralai import Mistral
import base64
from typing import List, Tuple, Dict
from typing import Any
from abc import ABC, abstractmethod

class MistralRequestStrategy(ABC):
    """
    Абстрактный клас для работы с API Mistral
    """

    @abstractmethod
    def send(self, *args: Any, **kwargs: Any) -> dict:
        pass

class TextRequest(MistralRequestStrategy):
    """
    Класс cnhfntubz отвечает за отправку текстовых запросов
    """
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = Mistral(api_key=self.api_key)
        
    def send(self, text: str, history: list = None,  model: str = "mistral-large-latest") -> dict:
        """
        Основной метод отправки текстового запроса к API Mistral
        """
        messages = []
        if history:
            messages.extend
        response = self.client.chat.complete(
            model=model,
            messages=[
                {
                    "role":"user",
                    "content": text
                }
            ]
        )
        #Формируем ответ в виде словаря для работы с историей чата 
        result = {
            "role": "assistant",
            "content": response.choices[0].message.content
        }

        return result
    
class ImageRequest(MistralRequestStrategy):
    """
    Класс отвечает за отправку запросов, включающих изображение.
    """
    def __init__(self, api_key: str) ->None:
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
    
    def send(self, text: str, image_path: str, model: str = "pixtral-12b-2409") -> dict:
        """
        Основной метод отравки мультимодального  запроса
        """
        # Получаем изображение в формате base64
        base64_image = self.__encode_image(image_path)
        # Формируем сообщение для чата
        messages = [
            {
                "role" : "user",
                "content" : [
                    {
                        "type" : "text",
                        "text" : text
                    },
                    {
                        "type" : "image_url",
                        "image_url" : f"data:image/jpeg;base64,{base64_image}"
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
            "role" : "assistant", 
            "content" : chat_response.choices[0].message.content
        }

        return result
    
# # text_request = TextRequest(api_key=MISTAL_API_KEY)
# image_request = ImageRequest(api_key=MISTAL_API_KEY)
# # text_response = text_request.send("Бонжур", model="mistral-large-latest")
# # print(text_response)
# image_response = image_request.send("что изображено на картинке", r"C:\Users\User-X\Desktop\Python\Request_to_Mistral\357d7153677451.593d686ecec5f.png", model= "pixtral-12b-2409")
# print(image_response)

class MistralRequesContext:
    """
    Контекст для работы со стратегиями запросов к Мистрал
    Реализуется паттерн Стратегия
    """
    def __init__(self, api_key: str)-> None:
        pass

class ChatFacade:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = {
            "text": ["mistral-large-latest"],
            "image": ["pixtral-12b-2409"]
        }
        self.set_request: TextRequest | ImageRequest = self.__set_request()
        self.model: str = self.__set_model()
        self.history = []

    def __set_request(self) -> TextRequest | ImageRequest:
        """
        Возвращает выбранный объект в зависимость от выбора пользователя
        """
        mode = input("Введите режим запроса (1 - текстовый, 2 - с изображением): ")

        if mode == "1":
            return TextRequest(api_key=self.api_key)
        elif mode == "2":
            return ImageRequest(api_key=self.api_key)
        else:
            raise ValueError("Неверный режим запроса")

    def __set_model(self) -> str:
        """
        Возвращается выбранную модель для запроса
        """
        model_type = 'text' if isinstance(self.set_request, TextRequest) else 'image'
        model = input(f"Выберите модель из списка {self.models[model_type]}: ")
        if model not in self.models[model_type]:
            raise ValueError('Неверная модель')
        return model

    def ask_question(self, text: str, image_path: str = None) -> dict:
        """
        Основной метод для отправки запроса
        """
        # Создаем сообщение пользователя
        user_message = {"role": "user", "content": text}
        # Получаем текущую историю
        current_history = [msg for _, msg in self.history]
        if image_path:
            response = self.set_request.send(text=text, image_path=image_path, model=self.model)
        else:
            response = self.set_request.send(text=text, model=self.model)
        # Обновляем историю
        self.history.append((text, user_message))
        self.history.append((text, response))
        return response

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
            image_path = None
            if isinstance(self.set_request, ImageRequest):
                image_path = input("Введите путь к изображению: ")
            response = self.ask_question(text=text, image_path=image_path if image_path else None)

            # Выводим последний ответ
            print(response)

chat_facade = ChatFacade(api_key=MISTAL_API_KEY)
chat_facade()

chat_facade = ChatFacade(api_key=MISTAL_API_KEY)
chat_facade()

