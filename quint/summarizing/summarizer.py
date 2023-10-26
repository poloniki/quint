import openai
import logging
import time


class TextSummarizer:
    def __init__(self, model="gpt-3.5-turbo", max_retries=5, retry_delay=1):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def summarize(self, text):
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": f'Summarize this text: "{text}"'}
                    ],
                )
                logging.info("OpenAI query successful.")
                return self.parse_response(response)
            except openai.error.OpenAIError as e:
                logging.error(f"OpenAI query error: {e}")
                if (
                    "is currently overloaded" in str(e)
                    and attempt < self.max_retries - 1
                ):
                    logging.info("Retrying...")
                    time.sleep(self.retry_delay)
                    continue
                raise ValueError("OpenAI query error: " + str(e)) from e
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                raise ValueError("Unexpected error: " + str(e)) from e

    @staticmethod
    def parse_response(response):
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logging.error(f"Error parsing OpenAI response: {e}")
            raise ValueError("Error parsing OpenAI response") from e


if __name__ == "__main__":
    summarizer = TextSummarizer()
    summary = summarizer.summarize("Your text here")
    print(summary)
