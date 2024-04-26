import tkinter as tk
from ollama_test import ChatBot
import json

class ChatWindow:
    def __init__(self, master):
        self.master = master
        master.title("Chat Window")

        # Create an instance of the ChatBot class
        self.chat_bot = ChatBot(model_name="llama3")

        self.chat_log = tk.Text(master, state='disabled')
        self.chat_log.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.entry_field = tk.Entry(master)
        self.entry_field.bind("<Return>", self.send_message)
        self.entry_field.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

    def send_message(self, event=None):
        message = self.entry_field.get()
        response = self.chat_bot.ask_question(message)  # Call ask_question on the chat_bot instance
        self.entry_field.delete(0, tk.END)
        self.display_message("User: " + message)
        self.display_message("Bot: " + response)

    def display_message(self, message):
        self.chat_log.configure(state='normal')
        if message.startswith("User:"):
            self.chat_log.insert(tk.END, message + '\n')
        elif message.startswith("Bot:"):
            self.chat_log.insert(tk.END, message + '\n')
        else:
            self.chat_log.insert(tk.END, "User: " + message + '\n')
        self.chat_log.configure(state='disabled')
        self.chat_log.see(tk.END)


    def load_questions_and_answers(self, file_path):
        """ Load training data from a JSON file and separate it into questions and answers.

        Args:
            file_path (str): The path to the JSON file containing the training data.

        Returns:
            tuple: Two lists, the first containing questions and the second containing answers.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            if not isinstance(data, list):
                raise ValueError("Data should be a list of dictionaries.")
            questions = [item['question'] for item in data if isinstance(item, dict)]
            answers = [item['answer'] for item in data if isinstance(item, dict)]
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")
            return [], []
        except json.JSONDecodeError:
            print("Error decoding JSON. Please check the file format.")
            return [], []
        except Exception as e:
            print(f"An error occurred: {e}")
            return [], []
        
        return questions, answers

def main():
    # Example usage
    file_path = 'data.json'
    root = tk.Tk()
    chat_window = ChatWindow(root)
    questions, answers = chat_window.load_questions_and_answers(file_path)
    root.mainloop()

if __name__ == "__main__":
    main()

def load_questions_and_answers(file_path):
    """ Load training data from a JSON file and separate it into questions and answers.

    Args:
        file_path (str): The path to the JSON file containing the training data.

    Returns:
        tuple: Two lists, the first containing questions and the second containing answers.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Data should be a list of dictionaries.")
        questions = [item['question'] for item in data if isinstance(item, dict)]
        answers = [item['answer'] for item in data if isinstance(item, dict)]
    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        return [], []
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file format.")
        return [], []
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []
    
    return questions, answers

# Example usage
file_path = 'data.json'
questions, answers = load_questions_and_answers(file_path)

# # Output the lists to verify
# print("Questions loaded:")
# for question in questions:
#     print(question)

# print("\nAnswers loaded:")
# for answer in answers:
#     print(answer)
